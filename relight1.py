import os
import numpy as np
from PIL import Image
import moderngl
import shutil
from tqdm import tqdm

class FilmicLightRenderer:

    def __init__(self, input_path, mask_path, albedo_path, normal_path, depth_path, output_base_path, copy_source=True):
        self.copy_source = copy_source
        self.ctx = None
        try:
            self.ctx = moderngl.create_standalone_context()
            print("OpenGL context created successfully.")
        except Exception as e:
            print(f"Failed to create OpenGL context: {e}")
            self.ctx = None

        self.window_size = (512, 512)

        # 深度/相机相关参数
        self.depth_path = depth_path
        self.depth_scale = 2.2       # 深度缩放，先从 2.0~3.0 试
        self.depth_bias = 0.05       # 防止 z=0
        self.depth_invert = True     # 如果你的深度图是“亮=近，暗=远”，这里设 True
        self.focal_uv = (1.35, 1.35) # 伪相机焦距，值越大透视越弱

        # 光源相对参考点的偏移（单位跟 z 同尺度）
        # 左、上、朝相机方向
        self.light_offset = (-1.00, -1.00, -1.35)

        # 光照参数
        self.ambient_strength = 0.52
        self.face_boost = 0.20
        self.atten_k1 = 0.12
        self.atten_k2 = 0.03

        self.depth_falloff = 0.80
        self.far_light_floor = 0.84

        self.depth_range = 1.55
        self.depth_back_scale = 1.8
        self.depth_back_gamma = 0.75
        self.depth_front_scale = 0.82

        self.face_detail_strength = 0.28
        self.bg_fill_strength = 0.10
        self.bg_global_lift = 0.035

        self.global_alpha = 0.82         # 最终叠加强一点

        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec3 in_position;
                in vec2 in_texcoord_0;
                out vec2 uv;
                void main() {
                    gl_Position = vec4(in_position, 1.0);
                    uv = in_texcoord_0;
                }
            ''',
            fragment_shader='''
                #version 330

                uniform sampler2D tex_input;
                uniform sampler2D tex_mask;
                uniform sampler2D tex_albedo;
                uniform sampler2D tex_normal;
                uniform sampler2D tex_depth;

                uniform vec3 light_color;
                uniform float blend_alpha;
                uniform float mask_factor;
                uniform bool use_tonemap;
                uniform int display_mode;

                uniform vec3 light_pos_ws;      // 世界/相机坐标系中的点光源位置
                uniform vec2 focal_uv;          // 伪相机焦距
                uniform float depth_scale;
                uniform float depth_bias;
                uniform bool depth_invert;

                uniform float ambient_strength;
                uniform float atten_k1;
                uniform float atten_k2;
                uniform float face_boost;
                uniform float face_detail_strength;
                uniform float bg_fill_strength;
                uniform float bg_global_lift;

                uniform float ref_depth_z;
                uniform float depth_falloff;
                uniform float far_light_floor;

                uniform float depth_range;
                uniform float depth_back_scale;
                uniform float depth_back_gamma;
                uniform float depth_front_scale;

                in vec2 uv;
                out vec4 f_color;

                vec3 aces(vec3 x) {
                    float a = 2.51;
                    float b = 0.03;
                    float c = 2.43;
                    float d = 0.59;
                    float e = 0.14;
                    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
                }

                vec3 decode_normal(vec3 raw_n) {
                    vec3 n = raw_n * 2.0 - 1.0;

                    // 很多 normal map 这里需要翻 Y，先试这版
                    n.y = -n.y;

                    // 防止局部 z 为负，出现“背向相机”的脏斑
                    n.z = abs(n.z);

                    return normalize(n);
                }
                
                float sample_depth(vec2 uv0) {
                    return texture(tex_depth, uv0).r;
                }

                float smooth_depth_5tap(vec2 uv0) {
                    vec2 texel = 1.0 / vec2(textureSize(tex_depth, 0));

                    float d0 = sample_depth(uv0) * 4.0;
                    float dl = sample_depth(clamp(uv0 + vec2(-texel.x, 0.0), vec2(0.0), vec2(1.0)));
                    float dr = sample_depth(clamp(uv0 + vec2( texel.x, 0.0), vec2(0.0), vec2(1.0)));
                    float du = sample_depth(clamp(uv0 + vec2(0.0, -texel.y), vec2(0.0), vec2(1.0)));
                    float dd = sample_depth(clamp(uv0 + vec2(0.0,  texel.y), vec2(0.0), vec2(1.0)));

                    return (d0 + dl + dr + du + dd) / 8.0;
                }
                // 用深度把每个像素恢复成相机坐标系里的3D点
                vec3 reconstruct_position(vec2 uv0, float raw_depth) {
                    float d = depth_invert ? (1.0 - raw_depth) : raw_depth;
                    float z = depth_bias + d * depth_scale;

                    // uv: [0,1] -> [-1,1]
                    float x_ndc = uv0.x * 2.0 - 1.0;
                    float y_ndc = 1.0 - uv0.y * 2.0;

                    float x = x_ndc * z / focal_uv.x;
                    float y = y_ndc * z / focal_uv.y;

                    return vec3(x, y, z);
                }

                void main() {
                    vec3 bg = pow(texture(tex_input, uv).rgb, vec3(2.2));
                    vec3 albedo = pow(texture(tex_albedo, uv).rgb, vec3(2.2));
                    vec3 norm_raw = texture(tex_normal, uv).rgb;
                    float raw_mask = texture(tex_mask, uv).r;
                    float raw_depth = texture(tex_depth, uv).r;
                    float depth_smooth = smooth_depth_5tap(uv);
                    vec3 P = reconstruct_position(uv, depth_smooth);

                    vec2 texel = 1.0 / vec2(textureSize(tex_normal, 0));
                    vec2 uv_min = texel * 0.5;
                    vec2 uv_max = vec2(1.0) - uv_min;

                    vec2 uv_l = clamp(uv + vec2(-texel.x, 0.0), uv_min, uv_max);
                    vec2 uv_r = clamp(uv + vec2( texel.x, 0.0), uv_min, uv_max);
                    vec2 uv_u = clamp(uv + vec2(0.0, -texel.y), uv_min, uv_max);
                    vec2 uv_d = clamp(uv + vec2(0.0,  texel.y), uv_min, uv_max);

                    vec3 normal_geom = decode_normal(norm_raw);
                    vec3 normal_l = decode_normal(texture(tex_normal, uv_l).rgb);
                    vec3 normal_r = decode_normal(texture(tex_normal, uv_r).rgb);
                    vec3 normal_u = decode_normal(texture(tex_normal, uv_u).rgb);
                    vec3 normal_d = decode_normal(texture(tex_normal, uv_d).rgb);

                    vec3 normal_shade = normalize(
                        normal_geom * 2.0 +
                        normal_l + normal_r + normal_u + normal_d
                    );

                    float normal_detail = (
                        (1.0 - dot(normal_geom, normal_l)) +
                        (1.0 - dot(normal_geom, normal_r)) +
                        (1.0 - dot(normal_geom, normal_u)) +
                        (1.0 - dot(normal_geom, normal_d))
                    ) * 0.5;

                    float normal_consistency =
                        (dot(normal_geom, normal_l) +
                        dot(normal_geom, normal_r) +
                        dot(normal_geom, normal_u) +
                        dot(normal_geom, normal_d)) * 0.25;

                    // 邻域一致性越高，说明这块法线越可信
                    float normal_conf = smoothstep(0.55, 0.92, normal_consistency);

                    // 正面朝向越弱，越容易是噪声/翻面区域
                    float facing_conf = smoothstep(0.08, 0.30, normal_shade.z);

                    float detail_conf = normal_conf * facing_conf;

                    normal_detail = smoothstep(0.02, 0.18, normal_detail);

                    // 软mask，避免脸和背景交界太硬
                    float raw_mask_soft = smoothstep(0.02, 0.98, raw_mask);

                    float face_core_mask = smoothstep(0.18, 0.92, raw_mask);
                    face_core_mask = face_core_mask * face_core_mask;

                    // 点光源方向
                    vec3 wi = light_pos_ws - P;
                    float dist = max(length(wi), 1e-5);
                    wi = wi / dist;

                    // 更柔的包裹漫反射
                    float NdotL = dot(normal_shade, wi);
                    float wrap = 0.50;   // 原来 0.65 太软，容易把折叠细节洗掉
                    float soft_diff = clamp((NdotL + wrap) / (1.0 + wrap), 0.0, 1.0);

                    float facing = max(dot(normal_shade, vec3(0.0, 0.0, 1.0)), 0.0);
                    float sculpt = mix(0.86, 1.14, pow(facing, 0.70));

                    float diff = soft_diff * sculpt;

                    float atten_dist = 1.0 / (1.0 + atten_k1 * dist + atten_k2 * dist * dist);

                    // ---------- 以主体为中心重映射 depth ----------
                    float z_rel = P.z - ref_depth_z;
                    float z_back = max(z_rel, 0.0);
                    float z_front = max(-z_rel, 0.0);

                    // 背景深度拉伸：让远处更容易“退后”
                    float z_back_norm = clamp(z_back / depth_range, 0.0, 1.0);
                    float z_back_remap = pow(z_back_norm, depth_back_gamma) * depth_back_scale;

                    // 前景压缩：避免主体内部层次被拉得太散
                    float z_front_norm = clamp(z_front / depth_range, 0.0, 1.0);
                    float z_front_remap = z_front_norm * depth_front_scale;

                    // depth01 表示“离主体往后退了多少”
                    float depth01 = clamp(z_back_remap, 0.0, 1.0);

                    // 主体深度层
                    float subject_depth_mask = exp(-3.0 * z_back_remap) * exp(-0.35 * z_front_remap);

                    // 背景深度衰减
                    float atten_depth = far_light_floor + (1.0 - far_light_floor) * exp(-depth_falloff * z_back_remap);

                    // 距离衰减只参与一部分，避免整图被压暗
                    float atten_mix = mix(1.0, atten_dist, 0.35);

                    // 最终主光
                    float key_light = diff * atten_mix * (0.40 + 0.60 * atten_depth);
                    float light_scalar = ambient_strength + key_light;

                    // 主体增强也主要从 depth 出，而不是只靠 mask
                    float face_gain_depth  = 0.28 * subject_depth_mask * face_core_mask * mask_factor;
                    float face_gain_near   = 0.14 * (1.0 - depth01) * face_core_mask * mask_factor;
                    float face_gain_face   = face_boost * face_core_mask * pow(facing, 0.7) * mask_factor;
                    float face_gain_detail = face_detail_strength * normal_detail * detail_conf * face_core_mask * mask_factor;

                    float local_boost = 1.0 + face_gain_depth + face_gain_near + face_gain_face + face_gain_detail;

                    vec3 lit_full = albedo * (light_scalar * light_color);
                    vec3 lit_face_boost = albedo * (light_scalar * local_boost * light_color);

                    // 局部细节区域再增加一点对比，让折叠感回来
                    float detail_contrast = 1.0 + 0.10 * normal_detail * detail_conf * raw_mask_soft * mask_factor * (0.5 + 0.5 * soft_diff);
                    detail_contrast = clamp(detail_contrast, 1.0, 1.10);

                    lit_face_boost *= detail_contrast;

                    // 背景退饱和 / 退对比，也只从 depth 出
                    float bg_depth_weight = (1.0 - subject_depth_mask) * (1.0 - raw_mask_soft);

                    float luma_full = dot(lit_full, vec3(0.299, 0.587, 0.114));
                    lit_full = mix(lit_full, vec3(luma_full), 0.10 * bg_depth_weight);

                    float luma_face = dot(lit_face_boost, vec3(0.299, 0.587, 0.114));
                    lit_face_boost = mix(lit_face_boost, vec3(luma_face), 0.04 * bg_depth_weight);

                    vec3 mid_gray = vec3(0.18);
                    lit_full = (lit_full - mid_gray) * mix(1.0, 0.82, bg_depth_weight) + mid_gray;
                    lit_face_boost = (lit_face_boost - mid_gray) * mix(1.0, 0.92, bg_depth_weight) + mid_gray;

                    // 最终更偏向主体版
                    vec3 lit_cinematic = mix(lit_full, lit_face_boost, 0.88);

                    // fill 只给背景，不给主体
                    float bg_fill = bg_global_lift + bg_fill_strength * bg_depth_weight;
                    vec3 source_fill = bg * bg_fill;
                    
                    
                    if (display_mode == 6) {
                        vec3 linear_result = bg + (lit_cinematic + source_fill) * blend_alpha;
                        vec3 final_color;
                        if (use_tonemap) {
                            final_color = aces(linear_result);
                            final_color = pow(final_color, vec3(1.0 / 2.2));
                        } else {
                            final_color = pow(max(linear_result, 0.0), vec3(1.0 / 2.2));
                        }
                        f_color = vec4(final_color, 1.0);
                    }
                    else if (display_mode == 7) {
                        vec3 linear_result = bg + (lit_full + source_fill) * blend_alpha;
                        vec3 final_color;
                        if (use_tonemap) {
                            final_color = aces(linear_result);
                            final_color = pow(final_color, vec3(1.0 / 2.2));
                        } else {
                            final_color = pow(max(linear_result, 0.0), vec3(1.0 / 2.2));
                        }
                        f_color = vec4(final_color, 1.0);
                    }
                    else {
                        if (display_mode == 1) f_color = vec4(pow(bg, vec3(1.0 / 2.2)), 1.0);
                        else if (display_mode == 2) f_color = vec4(vec3(raw_mask), 1.0);
                        else if (display_mode == 3) f_color = vec4(pow(albedo, vec3(1.0 / 2.2)), 1.0);
                        else if (display_mode == 4) f_color = vec4(norm_raw, 1.0);
                        else if (display_mode == 5) f_color = vec4(pow(max(lit_full, 0.0), vec3(1.0 / 2.2)), 1.0);
                        else f_color = vec4(0.0, 0.0, 0.0, 1.0);
                    }
                }
                '''
        )
        self.quad_vao = self.create_quad_vao()
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.renderbuffer(self.window_size, 4)]
        )
        self.init_env_map_cache()

        self.input_path = input_path
        self.mask_path = mask_path
        self.albedo_path = albedo_path
        self.normal_path = normal_path
        self.output_base_path = output_base_path

        self.display_mode = 6
        self.mouse_pos = (0.5, 0.5)
        self.use_tonemap = False
        self.PALETTE = [
            (1.0, 0.95, 0.8),
            (1.0, 0.2, 0.1),
            (0.1, 0.6, 1.0),
            (0.5, 1.0, 0.5),
            (1.0, 1.0, 1.0),
        ]
        self.palette_idx = 0
        self.current_color = np.array(self.PALETTE[0])
        self.mask_enabled = True
        self.mask_val_smooth = 1.0

        self.light_direction = (0.0, 0.0, 1.0)
        self.light_size = 0.5

        self.face_detail_strength = 0.28   # 面部法线细节增强
        self.bg_fill_strength = 0.10       # 背景补光强度
        self.bg_global_lift = 0.035        # 全背景基础提亮

    def read_depth_array(self, depth_file):
        depth = np.array(Image.open(depth_file), dtype=np.uint16).astype(np.float32) / 65535.0
        if self.depth_invert:
            depth = 1.0 - depth
        return depthz

    def estimate_light_from_depth(self, mask_file, depth_file):
        mask = np.array(Image.open(mask_file).convert('L'), dtype=np.float32) / 255.0
        depth = np.array(Image.open(depth_file), dtype=np.uint16).astype(np.float32) / 65535.0

        if self.depth_invert:
            depth = 1.0 - depth

        h, w = depth.shape

        ys, xs = np.where(mask > 0.1)
        if len(xs) == 0:
            u = 0.5
            v = 0.5
        else:
            u = float(xs.mean() / max(w - 1, 1))
            v = float(ys.mean() / max(h - 1, 1))

        xi = int(np.clip(round(u * (w - 1)), 0, w - 1))
        yi = int(np.clip(round(v * (h - 1)), 0, h - 1))

        z = self.depth_bias + depth[yi, xi] * self.depth_scale

        x_ndc = u * 2.0 - 1.0
        y_ndc = 1.0 - v * 2.0

        x = x_ndc * z / self.focal_uv[0]
        y = y_ndc * z / self.focal_uv[1]

        ref_point = np.array([x, y, z], dtype=np.float32)

        offset = np.array(self.light_offset, dtype=np.float32) * z
        light_pos_ws = ref_point + offset

        light_dir = light_pos_ws - ref_point
        norm = np.linalg.norm(light_dir) + 1e-8
        self.light_direction = tuple((light_dir / norm).tolist())

        return (u, v), light_pos_ws, z
    def init_env_map_cache(self):
        width, height = self.window_size
        x_coords = np.linspace(0, 2*np.pi, width, endpoint=False)
        y_coords = np.linspace(0, np.pi, height, endpoint=False)
        theta_grid, phi_grid = np.meshgrid(x_coords, y_coords)

        dir_x = np.sin(phi_grid) * np.cos(theta_grid)
        dir_y = np.cos(phi_grid)
        dir_z = np.sin(phi_grid) * np.sin(theta_grid)

        self.sample_dirs = np.stack([dir_x, dir_y, dir_z], axis=-1)

    def create_quad_vao(self):
        if self.ctx is None:
            return None
        vertices = np.array([
            -1.0, -1.0, 0.0, 0.0, 0.0,
             1.0, -1.0, 0.0, 1.0, 0.0,
             1.0,  1.0, 0.0, 1.0, 1.0,
            -1.0,  1.0, 0.0, 0.0, 1.0,
        ], dtype='f4')
        indices = np.array([0, 1, 2, 0, 2, 3], dtype='i4')
        vbo = self.ctx.buffer(vertices)
        ibo = self.ctx.buffer(indices)
        vao = self.ctx.vertex_array(
            self.prog,
            [(vbo, '3f 2f', 'in_position', 'in_texcoord_0')],
            ibo
        )
        return vao  
    def load_texture_2d(self, path):
        if self.ctx is None:
            return None
        img = Image.open(path).convert('RGBA')
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        texture = self.ctx.texture(img.size, 4, img.tobytes())
        texture.build_mipmaps()
        return texture
    
    def load_texture_depth(self, path):
        if self.ctx is None:
            return None

        img = Image.open(path)
        depth16 = np.array(img, dtype=np.uint16).astype(np.float32) / 65535.0
        depth16 = np.flipud(depth16).copy()

        texture = self.ctx.texture(
            depth16.shape[::-1],
            1,
            depth16.astype("f4").tobytes(),
            dtype="f4",
        )
        texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        texture.repeat_x = False
        texture.repeat_y = False
        return texture
    
    def process_single_image(self, input_file, mask_file, albedo_file, normal_file, depth_file, output_file, env_map_file, source_copy_file=None):
         tex_input = tex_mask = tex_albedo = tex_normal = tex_depth = None
         try:
             input_img = Image.open(input_file)
             input_width, input_height = input_img.size
             self.window_size = (input_width, input_height)
         except Exception as e:
             print(f"Failed to open input image {input_file}: {e}")
             self.window_size = (512, 512) #依据输入图片尺寸进行修改
        
         if self.copy_source and source_copy_file:
            try:
                os.makedirs(os.path.dirname(source_copy_file), exist_ok=True)
                shutil.copy2(input_file, source_copy_file)
            except Exception as e:
                print(f"Failed to copy source image to {source_copy_file}: {e}")
         elif self.copy_source:
                print(f"Source copy file path not provided, skipping copy for {input_file}")
         if self.ctx is None:
             self.save_light_environment_map(env_map_file)
             return True
         try:
             self.fbo = self.ctx.framebuffer(
                 color_attachments=[self.ctx.renderbuffer(self.window_size, 4)]
            )
             self.init_env_map_cache()
             self.quad_vao = self.create_quad_vao()

             tex_input = self.load_texture_2d(input_file)
             tex_mask = self.load_texture_2d(mask_file)
             tex_albedo = self.load_texture_2d(albedo_file)
             tex_normal = self.load_texture_2d(normal_file)
             tex_depth = self.load_texture_depth(depth_file)

             if not all([tex_input, tex_mask, tex_albedo, tex_normal, tex_depth]):
                 print(f"Failed to load one or more textures for {input_file}")
                 self.save_light_environment_map(env_map_file)
                 return True

             for tex in [tex_input, tex_mask, tex_albedo]:
                 tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

             tex_normal.filter = (moderngl.NEAREST, moderngl.NEAREST)
             tex_depth.filter = (moderngl.LINEAR, moderngl.LINEAR)

             tex_input.use(location=0)
             tex_mask.use(location=1)
             tex_albedo.use(location=2)
             tex_normal.use(location=3)
             tex_depth.use(location=4)

             self.prog['tex_input'].value = 0
             self.prog['tex_mask'].value = 1
             self.prog['tex_albedo'].value = 2
             self.prog['tex_normal'].value = 3
             self.prog['tex_depth'].value = 4

             target_col = np.array(self.PALETTE[self.palette_idx])
             self.current_color = self.current_color * 0.92 + target_col * 0.08
             target_mask = 1.0 if self.mask_enabled else 0.0
             self.mask_val_smooth = self.mask_val_smooth * 0.9 + target_mask * 0.1

             self.prog['light_color'].value = tuple(self.current_color)
             ref_uv, light_pos_ws, ref_depth_z = self.estimate_light_from_depth(mask_file, depth_file)

             self.prog['light_pos_ws'].value = tuple(light_pos_ws.tolist())
             self.prog['focal_uv'].value = self.focal_uv
             self.prog['depth_scale'].value = self.depth_scale
             self.prog['depth_bias'].value = self.depth_bias
             self.prog['depth_invert'].value = self.depth_invert

             self.prog['ambient_strength'].value = self.ambient_strength
             self.prog['atten_k1'].value = self.atten_k1
             self.prog['atten_k2'].value = self.atten_k2
             self.prog['face_boost'].value = self.face_boost
             self.prog['face_detail_strength'].value = self.face_detail_strength
             self.prog['bg_fill_strength'].value = self.bg_fill_strength
             self.prog['bg_global_lift'].value = self.bg_global_lift

   
             self.prog['blend_alpha'].value = self.global_alpha
             self.prog['mask_factor'].value = self.mask_val_smooth
             self.prog['use_tonemap'].value = self.use_tonemap
             self.prog['display_mode'].value = self.display_mode
             
             self.prog['ref_depth_z'].value = float(ref_depth_z)
             self.prog['depth_falloff'].value = self.depth_falloff
             self.prog['far_light_floor'].value = self.far_light_floor
             
             self.prog['depth_range'].value = self.depth_range
             self.prog['depth_back_scale'].value = self.depth_back_scale
             self.prog['depth_back_gamma'].value = self.depth_back_gamma
             self.prog['depth_front_scale'].value = self.depth_front_scale
             
             self.fbo.use()
             self.ctx.enable(moderngl.CULL_FACE)
             self.ctx.enable(moderngl.DEPTH_TEST)
             self.ctx.clear(0.0, 0.0, 0.0)

             self.quad_vao.render(moderngl.TRIANGLES)

             os.makedirs(os.path.dirname(output_file), exist_ok=True)
             self.save_framebuffer_as_image(output_file)

             os.makedirs(os.path.dirname(env_map_file), exist_ok=True)
             self.save_light_environment_map(env_map_file)

             return True
         except Exception as e:
            print(f"Error processing image {input_file}: {e}")
            try:
                os.makedirs(os.path.dirname(env_map_file), exist_ok=True)
                self.save_light_environment_map(env_map_file)
            except:
                pass
            return False
         finally:
             if tex_input:
                 tex_input.release()
             if tex_mask:
                 tex_mask.release()
             if tex_albedo:
                 tex_albedo.release()
             if tex_normal:
                 tex_normal.release()   
             if tex_depth:
                 tex_depth.release()    
    
    def batch_process(self):
        input_files = []
        for file in os.listdir(self.input_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_files.append(file)
        if not input_files:
            print("No input images found.")
            return
        print(f"Found {len(input_files)} input images. Starting processing...")
        success_count = 0
        fail_count = 0

        for filename in tqdm(input_files, desc="Processing images"):
            try:
                input_file = os.path.join(self.input_path, filename)
                base_name = os.path.splitext(filename)[0]
                ext = '.png'

                output_filename = base_name.replace('Source', 'Render') + ext
                output_file = os.path.join(self.output_base_path, 'Render', output_filename)
                env_map_filename = base_name.replace('Source', 'HDRI') + ext
                env_map_file = os.path.join(self.output_base_path, 'HDRI', env_map_filename)

                mask_filename = filename.replace('Source', 'Alpha') 
                albedo_filename = filename.replace('Source', 'BaseColor') 
                normal_filename = filename.replace('Source', 'Normal')      
                 
                depth_filename = filename.replace('Source', 'Depth')
                depth_file = os.path.join(self.depth_path, depth_filename)       
                mask_file = os.path.join(self.mask_path, mask_filename)
                albedo_file = os.path.join(self.albedo_path, albedo_filename)
                normal_file = os.path.join(self.normal_path, normal_filename) 
                source_copy_file = None
                if self.copy_source:
                    source_copy_filename = base_name.replace('_Source_', '_Source_') + ext
                    source_copy_file = os.path.join(self.output_base_path, 'Source', source_copy_filename)
                if self.process_single_image(
                    input_file,
                    mask_file,
                    albedo_file,
                    normal_file,
                    depth_file,
                    output_file,
                    env_map_file,
                    source_copy_file
                ):
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                print(f"Unexpected error processing {filename}: {e}")
                fail_count += 1     
        print(f"Processing completed: {success_count} succeeded, {fail_count} failed.")

    def save_framebuffer_as_image(self, filename):
        if self.ctx is None:
            return
        width, height = self.window_size
        if self.fbo:
            buffer = self.fbo.read(components=4)
        else:
            buffer = self.ctx.fbo.read(components=4)
        img = Image.frombuffer('RGBA', (width, height), buffer, 'raw', 'RGBA', 0, 1)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img.save(filename)
    
    def save_light_environment_map(self, filename=None):
        try:
            width, height = self.window_size
            sample_dirs = self.sample_dirs

            light_dir = np.array(self.light_direction)
            light_color = np.array(self.current_color)
            light_intensity = self.global_alpha
            light_size = self.light_size

            dot_products = np.einsum('ijk,k->ij', sample_dirs, light_dir)
            base_intensity = np.maximum(dot_products, 0.0)

            if light_size > 0:
                cos_angle = np.einsum('ijk,k->ij', sample_dirs, light_dir)
                angles = np.arccos(np.clip(cos_angle, -1.0, 1.0))

                angle_attenuation = np.exp(-angles*light_size*2.0)

                base_intensity *= angle_attenuation
            final_intensity = base_intensity * light_intensity
            env_colors = final_intensity[..., np.newaxis] * light_color

            env_img_array = np.clip(env_colors*255.0, 0, 255).astype(np.uint8)
            env_img = Image.fromarray(env_img_array, 'RGB')
            if filename:
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                env_img.save(filename)
            else:
                default_path = "products/default_env_map.png"
                os.makedirs(os.path.dirname(default_path), exist_ok=True)
                env_img.save(default_path)
                print(f"Environment map saved to default path: {default_path}")     
        except Exception as e:
            print(f"Failed to save environment map: {e}")      

def main():
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "test02_filter"))
    input_path = os.path.join(base_path, "Source")
    mask_path = os.path.join(base_path, "Alpha")
    albedo_path = os.path.join(base_path, "BaseColor")
    normal_path = os.path.join(base_path, "Normal")
    depth_path = os.path.join(base_path, "Depth")
    output_base_path = os.path.join(base_path, "output2")

    renderer = FilmicLightRenderer(
        input_path=input_path,
        mask_path=mask_path,
        albedo_path=albedo_path,
        normal_path=normal_path,
        depth_path=depth_path,
        output_base_path=output_base_path,
        copy_source=False,
    )

    renderer.batch_process()

if __name__ == "__main__":
    main()     
