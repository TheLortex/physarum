use rand::Rng;
use std::convert::TryInto;
use std::rc::Rc;
use wgpu::util::DeviceExt;

use super::gpu;

pub struct Game {
    gpu: Rc<gpu::Gpu>,
    compute_pipeline: wgpu::ComputePipeline,
    compute_bind_group: wgpu::BindGroup,
    input_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    pub image_buffer: wgpu::Buffer,
    width: usize,
    height: usize,
    compute_render_pipeline: wgpu::ComputePipeline,
    compute_render_bind_group: wgpu::BindGroup,
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

impl Game {
    pub fn new(gpu: Rc<gpu::Gpu>, width: usize, height: usize) -> Self {
        let mut compiler = shaderc::Compiler::new().unwrap();
        let cs_module = load_compute_shader(
            &gpu.device,
            &mut compiler,
            "conway.comp",
            include_str!("conway.comp"),
        );
        let cs_render_module = load_compute_shader(
            &gpu.device,
            &mut compiler,
            "conway_render.comp",
            include_str!("conway_render.comp"),
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
                contents: bytemuck::cast_slice(&[height]),
                usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
            });

        // initial state
        let data = {
            let mut data = vec![0 as u32; width * height];
            for pix in data.iter_mut() {
                *pix = if rand::thread_rng().gen_range((0.)..(1.0)) < 0.1 {
                    1
                } else {
                    0
                };
            }
            data
        };

        let input_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Input Buffer"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
            });

        let output_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Output Buffer"),
                contents: bytemuck::cast_slice(&vec![0 as u32; width * height]),
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
            });

        let image_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Image Buffer"),
                contents: bytemuck::cast_slice(&vec![0 as u32; width * height]),
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
            });

        let staging_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (width * height * 4) as u64,
            usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: false,
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
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
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

        // ##################""

        let compute_render_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute_bind_group"),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: height_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: image_buffer.as_entire_binding(),
                },
            ],
            layout: &compute_bind_group_layout,
        });

        let compute_render_bind_group_layout =
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

        let compute_render_pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Compute render pipeline layout"),
                    bind_group_layouts: &[&compute_render_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let compute_render_pipeline =
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Compute render Pipeline"),
                    layout: Some(&compute_render_pipeline_layout),
                    entry_point: "main",
                    module: &cs_render_module,
                });

        Self {
            gpu,
            compute_pipeline,
            compute_bind_group,
            input_buffer,
            output_buffer,
            staging_buffer,
            image_buffer,
            width,
            height,
            compute_render_pipeline,
            compute_render_bind_group,
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
            compute_pass.dispatch((self.width / 32) as u32, (self.height / 32) as u32, 1);
        }

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute render pass"),
            });
            compute_pass.set_pipeline(&self.compute_render_pipeline);
            compute_pass.set_bind_group(0, &self.compute_render_bind_group, &[]);
            compute_pass.dispatch((self.width / 32) as u32, (self.height / 32) as u32, 1);
        }

        self.gpu.queue.submit(std::iter::once(encoder.finish()));
    }

//    pub async fn get(&self) -> Vec<u8> {
//        let mut encoder = self
//            .gpu
//            .device
//            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
//        encoder.copy_buffer_to_buffer(
//            &self.output_buffer,
//            0,
//            &self.staging_buffer,
//            0,
//            (self.width * self.height * 4) as u64,
//        );
//        self.gpu.queue.submit(Some(encoder.finish()));
//
//        let slice = self.staging_buffer.slice(..);
//        let map = slice.map_async(wgpu::MapMode::Read);
//        self.gpu.device.poll(wgpu::Maintain::Wait);
//        map.await.unwrap();
//        let result = slice
//            .get_mapped_range()
//            .chunks_exact(4)
//            .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
//            .collect::<Vec<u32>>();
//
//        self.staging_buffer.unmap();
//
//        result
//    }

    pub fn swap(&self) {
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &self.input_buffer,
            0,
            (self.width * self.height * 4) as u64,
        );
        self.gpu.queue.submit(Some(encoder.finish()));
    }
}
