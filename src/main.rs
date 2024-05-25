use futures::executor::block_on;
use std::{rc::Rc, thread};
use winit::{event::*, event_loop::EventLoop, keyboard::KeyCode, window::WindowBuilder};
use winit_input_helper::WinitInputHelper;

const SIZE: usize = 2048;

mod conway;
mod gpu;
mod physarum;
mod renderer;

fn main() {
    let mut input = WinitInputHelper::new();
    env_logger::init();

    let event_loop = EventLoop::new().expect("could not create event loop");

    let window = &WindowBuilder::new().build(&event_loop).unwrap();
    // Since main can't be async, we're going to need to block
    let gpu = Rc::new(block_on(gpu::Gpu::new(&window)));

    let mut state = renderer::State::new(gpu.clone(), window.inner_size(), SIZE, SIZE);
    let mut game = physarum::Game::new(gpu.clone(), SIZE, SIZE);

    let now = std::time::Instant::now();
    let mut frames_count = 0;

    event_loop
        .run(move |event, control_flow| {
            if input.update(&event) {
                if input.close_requested() {
                    let elapsed = now.elapsed().as_secs_f32();
                    println!(
                        "Time: {:?} / Frames: {:?} / FPS: {:?}",
                        elapsed,
                        frames_count,
                        (frames_count as f32) / elapsed
                    );
                    control_flow.exit();
                }
                if input.key_pressed(KeyCode::Escape) {
                    control_flow.exit();
                }

                if input.mouse_held(MouseButton::Left) {
                    let (dx, dy) = input.mouse_diff();
                    state.pan(dx, dy);
                }
            }

            match event {
                Event::WindowEvent {
                    ref event,
                    window_id,
                } if window_id == window.id() => {
                    if !state.input(event) {
                        match event {
                            WindowEvent::MouseWheel { delta, .. } => {
                                let v = match delta {
                                    winit::event::MouseScrollDelta::LineDelta(_, y) => *y,
                                    winit::event::MouseScrollDelta::PixelDelta(px) => px.y as f32,
                                };

                                state.zoom(v);
                            }
                            WindowEvent::Resized(physical_size) => {
                                log::info!("Resizer");
                                state.resize(*physical_size);
                            }
                            WindowEvent::ScaleFactorChanged { .. } => {
                                // new_inner_size is &&mut so we have to dereference it twice
                                //  state.resize(**new_inner_size);
                            }
                            WindowEvent::RedrawRequested => {
                                log::info!("RedrawRequested");
                                state.update();
                                match state.render() {
                                    Ok(_) => {
                                        frames_count += 1;
                                    }
                                    // Recreate the swap_chain if lost
                                    Err(wgpu::SurfaceError::Lost) => {
                                        log::warn!("Lost");
                                        state.resize(state.size)
                                    }
                                    // The system is out of memory, we should probably quit
                                    Err(wgpu::SurfaceError::OutOfMemory) => control_flow.exit(),
                                    // All other errors (Outdated, Timeout) should be resolved by the next frame
                                    Err(e) => eprintln!("{:?}", e),
                                }

                                log::info!("RedrawRequested done");
                                game.step();
                                log::info!("Game step done");

                                {
                                    let layout = wgpu::ImageDataLayout {
                                        offset: 0,
                                        bytes_per_row: Some(4 * SIZE as u32),
                                        rows_per_image: Some(SIZE as u32),
                                    };

                                    let texture_size = wgpu::Extent3d {
                                        width: SIZE as u32,
                                        height: SIZE as u32,
                                        depth_or_array_layers: 1,
                                    };

                                    let mut encoder = gpu.device.create_command_encoder(
                                        &wgpu::CommandEncoderDescriptor { label: None },
                                    );
                                    encoder.copy_buffer_to_texture(
                                        wgpu::ImageCopyBuffer {
                                            buffer: &game.buffers.render,
                                            layout,
                                        },
                                        wgpu::ImageCopyTexture {
                                            texture: &state.diffuse_texture,
                                            mip_level: 0,
                                            origin: wgpu::Origin3d::ZERO,
                                            aspect: wgpu::TextureAspect::All,
                                        },
                                        texture_size,
                                    );
                                    gpu.queue.submit(Some(encoder.finish()));
                                    log::info!("Copy done");
                                }

                                window.request_redraw();
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        })
        .expect("success");
}
