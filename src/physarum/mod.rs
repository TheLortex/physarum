use rand::Rng;
use std::rc::Rc;
use wgpu::util::DeviceExt;

use super::gpu;

pub struct Game {
    gpu: Rc<gpu::Gpu>,
    simulation_pipeline: wgpu::ComputePipeline,
    simulation_bind_group: wgpu::BindGroup,
    diffusion_pipeline: wgpu::ComputePipeline,
    diffusion_bind_group: wgpu::BindGroup,
    pub buffers: GameBuffers,
    width: usize,
    height: usize,
}

fn load_compute_shader(
    device: &wgpu::Device,
    compiler: &mut shaderc::Compiler,
    name: &str,
    source: &str,
) -> wgpu::ShaderModule {
    let cs_spirv = compiler
        .compile_into_spirv(source, shaderc::ShaderKind::Compute, name, "main", None)
        .unwrap();
    let cs_data = wgpu::util::make_spirv(cs_spirv.as_binary_u8());
    device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: Some("Compute Shader"),
        source: cs_data,
        flags: wgpu::ShaderFlags::default(),
    })
}

const N_TRACERS: usize = 2_000_000;

pub struct GameBuffers {
    size: wgpu::Buffer,
    state: wgpu::Buffer,
    image_input: wgpu::Buffer,
    image_output: wgpu::Buffer,
    pub render: wgpu::Buffer,
}

impl GameBuffers {
    fn new(gpu: &gpu::Gpu, width: usize, height: usize) -> Self {
        let size = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Height Buffer"),
                contents: bytemuck::cast_slice(&[height as u32, width as u32]),
                usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
            });

        // initial state
        let data = {
            let mut data = Vec::new();
            let mut rng = rand::thread_rng();
            for _ in 0..N_TRACERS {
                data.push(Tracer {
                    x: rng.gen_range(10..(width - 10)) as f32,
                    y: rng.gen_range(10..(height - 10)) as f32,
                    angle: rng.gen_range((0.)..(2. * 3.14)),
                })
            }
            data
        };

        let state = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("state Buffer"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
            });

        let image_input = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Image Buffer"),
                contents: bytemuck::cast_slice(&vec![0 as f32; width * height]),
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
            });

        let image_output = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("image output buffer"),
            mapped_at_creation: false,
            size: (width * height * 4) as u64,
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
        });

        let render = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render buffer"),
            mapped_at_creation: false,
            size: (width * height * 4) as u64,
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
        });

        GameBuffers {
            size,
            state,
            image_input,
            image_output,
            render,
        }
    }
}

use bytemuck::{Pod, Zeroable};
#[repr(C)]
#[derive(Copy, Clone, Zeroable, Pod)]
struct Tracer {
    x: f32,
    y: f32,
    angle: f32,
}

impl Game {
    fn get_diffusion_shader(
        gpu: &gpu::Gpu,
        buffers: &GameBuffers,
    ) -> (wgpu::ComputePipeline, wgpu::BindGroup) {
        let mut compiler = shaderc::Compiler::new().unwrap();

        let cs_module = load_compute_shader(
            &gpu.device,
            &mut compiler,
            "diffusion.comp",
            include_str!("diffusion.comp"),
        );

        let compute_bind_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("compute_bind_group_layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStage::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStage::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStage::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStage::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let compute_pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Compute pipeline layout"),
                    bind_group_layouts: &[&compute_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let compute_pipeline =
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Compute Pipeline"),
                    layout: Some(&compute_pipeline_layout),
                    entry_point: "main",
                    module: &cs_module,
                });

        let compute_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute_bind_group"),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.size.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.image_input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.image_output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.render.as_entire_binding(),
                },
            ],
            layout: &compute_bind_group_layout,
        });

        return (compute_pipeline, compute_bind_group);
    }

    fn get_simulation_shader(
        gpu: &gpu::Gpu,
        buffers: &GameBuffers,
    ) -> (wgpu::ComputePipeline, wgpu::BindGroup) {
        let mut compiler = shaderc::Compiler::new().unwrap();

        let cs_module = load_compute_shader(
            &gpu.device,
            &mut compiler,
            "physarum.comp",
            include_str!("physarum.comp"),
        );

        let compute_bind_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("compute_bind_group_layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStage::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStage::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStage::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let compute_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute_bind_group"),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.size.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.state.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.image_input.as_entire_binding(),
                },
            ],
            layout: &compute_bind_group_layout,
        });

        let compute_pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Compute pipeline layout"),
                    bind_group_layouts: &[&compute_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let compute_pipeline =
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Compute Pipeline"),
                    layout: Some(&compute_pipeline_layout),
                    entry_point: "main",
                    module: &cs_module,
                });

        return (compute_pipeline, compute_bind_group);
    }

    pub fn new(gpu: Rc<gpu::Gpu>, width: usize, height: usize) -> Self {
        let buffers = GameBuffers::new(&gpu, width, height);

        let (simulation_pipeline, simulation_bind_group) =
            Self::get_simulation_shader(&gpu, &buffers);

        let (diffusion_pipeline, diffusion_bind_group) = Self::get_diffusion_shader(&gpu, &buffers);

        Self {
            gpu,
            simulation_pipeline,
            simulation_bind_group,
            diffusion_pipeline,
            diffusion_bind_group,
            buffers,
            width,
            height,
        }
    }

    pub fn step(&self) {
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Compute Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute pass"),
            });
            compute_pass.set_pipeline(&self.simulation_pipeline);
            compute_pass.set_bind_group(0, &self.simulation_bind_group, &[]);
            compute_pass.dispatch((N_TRACERS / 32) as u32, 1, 1);
        }
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute pass"),
            });
            compute_pass.set_pipeline(&self.diffusion_pipeline);
            compute_pass.set_bind_group(0, &self.diffusion_bind_group, &[]);
            compute_pass.dispatch((self.width / 32) as u32, (self.height / 32) as u32, 1);
        }

        encoder.copy_buffer_to_buffer(
            &self.buffers.image_output,
            0,
            &self.buffers.image_input,
            0,
            (self.width * self.height * 4) as u64,
        );

        self.gpu.queue.submit(std::iter::once(encoder.finish()));
    }
}
