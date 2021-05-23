use rand::Rng;
use std::rc::Rc;
use wgpu::util::DeviceExt;

use super::gpu;

pub struct Game {
    gpu: Rc<gpu::Gpu>,
    compute_pipeline: wgpu::ComputePipeline,
    compute_bind_group: wgpu::BindGroup,
    state_buffer: wgpu::Buffer,
    pub image_buffer: wgpu::Buffer,
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

const N_TRACERS: usize = 100;

use bytemuck::{Pod, Zeroable};
#[repr(C)]
#[derive(Copy, Clone, Zeroable, Pod)]
struct Tracer {
    x: f32,
    y: f32,
    angle: f32,
}

impl Game {
    pub fn new(gpu: Rc<gpu::Gpu>, width: usize, height: usize) -> Self {
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

        let height_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Height Buffer"),
                contents: bytemuck::cast_slice(&[height, width]),
                usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
            });

        // initial state
        let data = {
            let mut data = Vec::new();
            let mut rng = rand::thread_rng();
            for _ in 0..N_TRACERS {
                data.push(Tracer {
                    x: rng.gen_range(10..(width-10)) as f32,
                    y: rng.gen_range(10..(height-10)) as f32,
                    angle: rng.gen_range((0.)..(2.*3.14))
                })
            }
            data
        };

        let state_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("state Buffer"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
            });

        let image_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Image Buffer"),
                contents: bytemuck::cast_slice(&vec![0 as u32; width * height]),
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
            });

        let compute_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute_bind_group"),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: height_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: image_buffer.as_entire_binding(),
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

        Self {
            gpu,
            compute_pipeline,
            compute_bind_group,
            state_buffer,
            image_buffer,
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
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
            compute_pass.dispatch((N_TRACERS / 32) as u32, 1, 1);
        }

        self.gpu.queue.submit(std::iter::once(encoder.finish()));
    }
}
