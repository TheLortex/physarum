use std::rc::Rc;
use wgpu::{util::DeviceExt, BlendState, PipelineCompilationOptions};
use winit::event::WindowEvent;

use super::gpu;

pub struct State<'surface> {
    gpu: Rc<gpu::Gpu<'surface>>,
    sc_desc: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    camera: Camera,
    uniforms: Uniforms,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    diffuse_bind_group: wgpu::BindGroup,
    pub diffuse_texture: wgpu::Texture,
    pub size: winit::dpi::PhysicalSize<u32>,
    width: usize,
    height: usize,

    zoom: f32,
    ofsx: f32,
    ofsy: f32,
}

fn max(a: f32, b: f32) -> f32 {
    if a < b {
        b
    } else {
        a
    }
}

fn min(a: f32, b: f32) -> f32 {
    if a > b {
        b
    } else {
        a
    }
}

impl<'surface> State<'surface> {
    pub fn new(
        gpu: Rc<gpu::Gpu<'surface>>,
        physicalsize: winit::dpi::PhysicalSize<u32>,
        width: usize,
        height: usize,
    ) -> Self {
        log::info!("Creating State");
        let format = wgpu::TextureFormat::Bgra8Unorm;

        let sc_desc = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: physicalsize.width,
            height: physicalsize.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            desired_maximum_frame_latency: 2,
            view_formats: vec![format],
        };
        gpu.surface.configure(&gpu.device, &sc_desc);

        let vs_src = include_str!("shader.vert");
        let fs_src = include_str!("shader.frag");

        let mut compiler = shaderc::Compiler::new().unwrap();
        let vs_spirv = compiler
            .compile_into_spirv(
                vs_src,
                shaderc::ShaderKind::Vertex,
                "shader.vert",
                "main",
                None,
            )
            .unwrap();
        let fs_spirv = compiler
            .compile_into_spirv(
                fs_src,
                shaderc::ShaderKind::Fragment,
                "shader.frag",
                "main",
                None,
            )
            .unwrap();

        let vs_data = wgpu::util::make_spirv(vs_spirv.as_binary_u8());
        let fs_data = wgpu::util::make_spirv(fs_spirv.as_binary_u8());

        let vs_module = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Vertex Shader"),
                source: vs_data,
            });
        let fs_module = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Fragment Shader"),
                source: fs_data,
            });

        let camera = Camera {
            ofsx: 0.,
            ofsy: 0.,
            zoom: 1.,
            aspect: (physicalsize.height as f32) / (physicalsize.width as f32),
        };

        // UNIFORM

        let mut uniforms = Uniforms::new();
        uniforms.update_view_proj(&camera);

        let uniform_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Uniform Buffer"),
                contents: bytemuck::cast_slice(&[uniforms]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let uniform_bind_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                    label: Some("uniform_bind_group_layout"),
                });
        let uniform_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
            label: Some("uniform_bind_group"),
        });

        let texture_size = wgpu::Extent3d {
            width: width as u32,
            height: height as u32,
            depth_or_array_layers: 1,
        };

        let diffuse_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            // All textures are stored as 3D, we represent our 2D texture
            // by setting depth to 1.
            size: texture_size,
            mip_level_count: 1, // We'll talk about this a little later
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            // SAMPLED tells wgpu that we want to use this texture in shaders
            // COPY_DST means that we want to copy data to this texture
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            label: Some("diffuse_texture"),
            view_formats: &[],
        });

        //let diffuse_rgba = {
        //    let mut data = vec![0x00; SIZE * SIZE * 4];
        //    for pix in data.iter_mut() {
        //        *pix = rand::random::<u8>();
        //    }
        //    data
        //};
        //
        //queue.write_texture(
        //    // Tells wgpu where to copy the pixel data
        //    wgpu::TextureCopyView {
        //        texture: &diffuse_texture,
        //        mip_level: 0,
        //        origin: wgpu::Origin3d::ZERO,
        //    },
        //    // The actual pixel data
        //    &diffuse_rgba,
        //    // The layout of the texture
        //    wgpu::TextureDataLayout {
        //        offset: 0,
        //        bytes_per_row: 4 * SIZE as u32,
        //        rows_per_image: SIZE as u32,
        //    },
        //    texture_size,
        //);

        let diffuse_texture_view =
            diffuse_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let diffuse_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let texture_bind_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                            count: None,
                        },
                    ],
                    label: Some("texture_bind_group_layout"),
                });

        let diffuse_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_sampler),
                },
            ],
            label: Some("diffuse_bind_group"),
        });

        // RENDER PIPELINE
        let render_pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Render Pipeline Layout"),
                    bind_group_layouts: &[&uniform_bind_group_layout, &texture_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let render_pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Render Pipeline"),
                layout: Some(&render_pipeline_layout),
                multiview: None,
                vertex: wgpu::VertexState {
                    module: &vs_module,
                    entry_point: "main", // 1.
                    buffers: &[],        // 2.
                    compilation_options: PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    // 3.
                    module: &fs_module,
                    entry_point: "main",
                    targets: &[Some(wgpu::ColorTargetState {
                        // 4.
                        format: sc_desc.format,
                        blend: Some(BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList, // 1.
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw, // 2.
                    cull_mode: Some(wgpu::Face::Back),
                    // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None, // 1.
                multisample: wgpu::MultisampleState {
                    count: 1,                         // 2.
                    mask: !0,                         // 3.
                    alpha_to_coverage_enabled: false, // 4.
                },
            });

        Self {
            size: physicalsize,
            gpu,
            sc_desc,
            render_pipeline,
            camera,
            uniforms,
            uniform_buffer,
            uniform_bind_group,
            diffuse_bind_group,
            diffuse_texture,
            width,
            height,
            zoom: 1.,
            ofsx: 0.,
            ofsy: 0.,
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        log::info!("Resizing to {:?}", new_size);
        self.size = new_size;
        self.sc_desc.width = self.size.width;
        self.sc_desc.height = self.size.height;
        self.gpu.surface.configure(&self.gpu.device, &self.sc_desc);
    }

    pub fn input(&mut self, _event: &WindowEvent) -> bool {
        false
    }

    pub fn update(&mut self) {
        log::info!("Updating");
        self.camera = Camera {
            ofsx: self.ofsx,
            ofsy: self.ofsy,
            zoom: self.zoom,
            aspect: (self.size.height as f32) / (self.size.width as f32),
        };
        self.uniforms.update_view_proj(&self.camera);
        self.gpu.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );
    }

    pub fn zoom(&mut self, delta: f32) {
        log::info!("Zooming by {}", delta);
        self.zoom += delta / 3.;

        if self.zoom < 0.95 {
            self.zoom = 0.95;
        } else if self.zoom > 10. {
            self.zoom = 10.
        }

        self.pan(0., 0.);
    }

    pub fn pan(&mut self, dx: f32, dy: f32) {
        self.ofsx += 4. * (dx / (self.width as f32)) / self.camera.zoom;
        self.ofsy -= 4. * (dy / (self.height as f32)) / self.camera.zoom;

        let zoomd = 0.9 / self.camera.zoom;
        self.ofsx = max(-1. + zoomd, self.ofsx);
        self.ofsx = min(1. - zoomd, self.ofsx);
        self.ofsy = max(-1. + zoomd, self.ofsy);
        self.ofsy = min(1. - zoomd, self.ofsy);
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        log::info!("Rendering");
        let frame = self.gpu.surface.get_current_texture()?;

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            render_pass.set_bind_group(1, &self.diffuse_bind_group, &[]);

            render_pass.draw(0..6, 0..1);
        }

        log::info!("Submitting");
        // submit will accept anything that implements IntoIter
        self.gpu.queue.submit(std::iter::once(encoder.finish()));
        log::info!("Submitted");

        frame.present();

        Ok(())
    }

    pub fn write(&self, fb: &[u8]) {
        self.gpu.queue.write_texture(
            // Tells wgpu where to copy the pixel data
            wgpu::ImageCopyTexture {
                texture: &self.diffuse_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            // The actual pixel data
            fb,
            // The layout of the texture
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * self.width as u32),
                rows_per_image: Some(self.height as u32),
            },
            wgpu::Extent3d {
                width: self.width as u32,
                height: self.height as u32,
                depth_or_array_layers: 1,
            },
        );
    }
}

// CAMERA

struct Camera {
    ofsx: f32,
    ofsy: f32,
    zoom: f32,
    aspect: f32,
}

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

impl Camera {
    fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        let translation = cgmath::Matrix4::from_translation((self.ofsx, self.ofsy, 0.).into());

        let aspect_ratio = if self.aspect > 1. {
            cgmath::Matrix4::from_nonuniform_scale(1., 1. / self.aspect, 1.)
        } else {
            cgmath::Matrix4::from_nonuniform_scale(self.aspect, 1., 1.)
        };
        let zoom = cgmath::Matrix4::from_scale(self.zoom);

        return OPENGL_TO_WGPU_MATRIX * zoom * translation * aspect_ratio;
    }
}

// We need this for Rust to store our data correctly for the shaders
#[repr(C)]
// This is so we can store this in a buffer
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    // We can't use cgmath with bytemuck directly so we'll have
    // to convert the Matrix4 into a 4x4 f32 array
    view_proj: [[f32; 4]; 4],
}

impl Uniforms {
    fn new() -> Self {
        use cgmath::SquareMatrix;
        Self {
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix().into();
    }
}
