use rand::Rng;
use std::rc::Rc;
use std::sync::Arc;
use wgpu::{util::DeviceExt, PipelineCompilationOptions};
use zerocopy::AsBytes;

use super::gpu;

pub mod settings;

#[repr(C)]
#[derive(Copy, Clone, zerocopy::AsBytes)]
struct Params {
    speed: f32,
    sensor_angle: f32,
    sensor_distance: f32,
    sensor_size: u32,
    rotation_speed: f32,
    decay_ratio: f32,
    diffusion_ratio: f32,
    entropy: f32,
    width: u32,
    height: u32,
    time: u32,
}

impl Params {
    fn size() -> usize {
        std::mem::size_of::<Self>()
    }

    fn of_settings(width: u32, height: u32, time: u32, settings: settings::Settings) -> Self {
        Self {
            width,
            height,
            time,
            speed: settings.speed,
            sensor_angle: settings.sensor_angle,
            sensor_distance: settings.sensor_distance,
            sensor_size: settings.sensor_size,
            rotation_speed: settings.rotation_speed,
            decay_ratio: settings.decay_ratio,
            diffusion_ratio: settings.diffusion_ratio,
            entropy: settings.entropy,
        }
    }
}

use std::sync::Mutex;

pub struct Game<'surface> {
    gpu: Rc<gpu::Gpu<'surface>>,
    simulation_pipeline: wgpu::ComputePipeline,
    simulation_bind_group: wgpu::BindGroup,
    diffusion_pipeline: wgpu::ComputePipeline,
    diffusion_bind_group: wgpu::BindGroup,
    pub buffers: GameBuffers,
    height: u32,
    width: u32,
    time: u32,
    settings: Arc<Mutex<settings::Settings>>,
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
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Compute Shader"),
        source: cs_data,
    })
}

const N_TRACERS: usize = 2_000_000;

pub struct GameBuffers {
    state: wgpu::Buffer,
    image_input: wgpu::Buffer,
    image_input_hue: wgpu::Buffer,
    image_output: wgpu::Buffer,
    image_output_hue: wgpu::Buffer,
    pub render: wgpu::Buffer,
}

impl GameBuffers {
    fn new(gpu: &gpu::Gpu, width: usize, height: usize) -> Self {
        log::info!("Creating buffers");
        // initial state
        let data = {
            let mut data = Vec::new();
            let mut rng = rand::thread_rng();
            for _ in 0..N_TRACERS {
                let angle = rng.gen_range((0.)..(2. * 3.14)) as f32;
                let distance = rng.gen_range((100.)..(700.)) as f32;
                let x = (width / 2) as f32 + distance * angle.sin();
                let y = (height / 2) as f32 + distance * angle.cos();
                let hue = rng.gen_range((0.)..(1.)) as f32;

                data.push(Tracer { x, y, angle, hue })
            }
            data
        };

        let state = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("state Buffer"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let image_input = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Image Buffer"),
                contents: bytemuck::cast_slice(&vec![0 as f32; width * height]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let image_input_hue = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Image Buffer"),
                contents: bytemuck::cast_slice(&vec![0 as f32; width * height]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let image_output = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("image output buffer"),
            mapped_at_creation: false,
            size: (width * height * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let image_output_hue = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("image output hue buffer"),
            mapped_at_creation: false,
            size: (width * height * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let render = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render buffer"),
            mapped_at_creation: false,
            size: (width * height * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        GameBuffers {
            state,
            image_input,
            image_input_hue,
            image_output,
            image_output_hue,
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
    hue: f32,
}

impl<'surface> Game<'surface> {
    fn get_diffusion_shader(
        gpu: &gpu::Gpu<'surface>,
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
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                    push_constant_ranges: &[wgpu::PushConstantRange {
                        range: 0..(Params::size() as u32),
                        stages: wgpu::ShaderStages::COMPUTE,
                    }],
                });

        let compute_pipeline =
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Compute Pipeline"),
                    layout: Some(&compute_pipeline_layout),
                    entry_point: "main",
                    module: &cs_module,
                    compilation_options: PipelineCompilationOptions::default(),
                });

        let compute_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute_bind_group"),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.image_input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.image_output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.render.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.image_input_hue.as_entire_binding(),
                },
            ],
            layout: &compute_bind_group_layout,
        });

        return (compute_pipeline, compute_bind_group);
    }

    fn get_simulation_shader(
        gpu: &gpu::Gpu<'surface>,
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
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                    resource: buffers.state.as_entire_binding(),
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
                    resource: buffers.image_output_hue.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.image_input_hue.as_entire_binding(),
                },
            ],
            layout: &compute_bind_group_layout,
        });

        let compute_pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Compute pipeline layout"),
                    bind_group_layouts: &[&compute_bind_group_layout],
                    push_constant_ranges: &[wgpu::PushConstantRange {
                        range: 0..(Params::size() as u32),
                        stages: wgpu::ShaderStages::COMPUTE,
                    }],
                });

        let compute_pipeline =
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Compute Pipeline"),
                    layout: Some(&compute_pipeline_layout),
                    entry_point: "main",
                    module: &cs_module,
                    compilation_options: PipelineCompilationOptions::default(),
                });

        return (compute_pipeline, compute_bind_group);
    }

    pub fn new(gpu: Rc<gpu::Gpu<'surface>>, width: usize, height: usize) -> Self {
        let buffers = GameBuffers::new(&gpu, width, height);

        let (simulation_pipeline, simulation_bind_group) =
            Self::get_simulation_shader(&gpu, &buffers);

        let (diffusion_pipeline, diffusion_bind_group) = Self::get_diffusion_shader(&gpu, &buffers);

        let settings = Arc::new(Mutex::new(settings::Settings::default()));

        let _settings_ref = settings.clone();
        /*let _handle = settings::SettingsManager::new(
            move |new_value| {
                let mut locked = settings_ref.lock().unwrap();
                *locked = new_value;
            },
            settings::Settings::default(),
        );*/

        Self {
            gpu,
            simulation_pipeline,
            simulation_bind_group,
            diffusion_pipeline,
            diffusion_bind_group,
            buffers,
            settings,
            width: width as u32,
            height: height as u32,
            time: 0,
        }
    }

    pub fn step(&mut self) {
        self.time += 1;

        let params = {
            let settings = self.settings.lock().unwrap();
            Params::of_settings(self.width, self.height, self.time, *settings)
        };

        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Compute Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute simulation pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.simulation_pipeline);
            compute_pass.set_bind_group(0, &self.simulation_bind_group, &[]);
            compute_pass.set_push_constants(0, params.as_bytes());
            compute_pass.dispatch_workgroups((N_TRACERS / 256) as u32, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &self.buffers.image_output,
            0,
            &self.buffers.image_input,
            0,
            (self.width * self.height * 4) as u64,
        );

        encoder.copy_buffer_to_buffer(
            &self.buffers.image_output_hue,
            0,
            &self.buffers.image_input_hue,
            0,
            (self.width * self.height * 4) as u64,
        );

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute diffusion pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.diffusion_pipeline);
            compute_pass.set_bind_group(0, &self.diffusion_bind_group, &[]);
            compute_pass.set_push_constants(0, params.as_bytes());
            compute_pass.dispatch_workgroups(self.width / 16, self.height / 16, 1);
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
