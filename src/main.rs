use futures::executor::block_on;
use std::rc::Rc;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
use winit_input_helper::WinitInputHelper;

const SIZE: usize = 2048;

mod conway;
mod physarum;
mod gpu;
mod renderer;

fn main() {
    let mut input = WinitInputHelper::new();
    env_logger::init();

    let event_loop = EventLoop::new();

    let window = WindowBuilder::new().build(&event_loop).unwrap();
    // Since main can't be async, we're going to need to block
    let gpu = Rc::new(block_on(gpu::Gpu::new(&window)));

    let mut state = renderer::State::new(gpu.clone(), window.inner_size(), SIZE, SIZE);
    let mut game = physarum::Game::new(gpu.clone(), SIZE, SIZE);

    let now = std::time::Instant::now();
    let mut frames_count = 0;

    event_loop.run(move |event, _, control_flow| {
        if input.update(&event) {
            if input.quit() {
                let elapsed = now.elapsed().as_secs_f32();
                println!(
                    "Time: {:?} / Frames: {:?} / FPS: {:?}",
                    elapsed,
                    frames_count,
                    (frames_count as f32) / elapsed
                );
                *control_flow = ControlFlow::Exit
            }
            if input.key_pressed(VirtualKeyCode::Escape) {
                *control_flow = ControlFlow::Exit
            }

            if input.mouse_held(0) {
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
                            state.resize(*physical_size);
                        }
                        WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                            // new_inner_size is &&mut so we have to dereference it twice
                            state.resize(**new_inner_size);
                        }
                        _ => {}
                    }
                }
            }
            Event::RedrawRequested(_) => {
                state.update();
                match state.render() {
                    Ok(_) => {
                        frames_count += 1;
                    }
                    // Recreate the swap_chain if lost
                    Err(wgpu::SwapChainError::Lost) => state.resize(state.size),
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SwapChainError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    // All other errors (Outdated, Timeout) should be resolved by the next frame
                    Err(e) => eprintln!("{:?}", e),
                }

                game.step();

                {
                    let layout = wgpu::TextureDataLayout {
                        offset: 0,
                        bytes_per_row: 4 * SIZE as u32,
                        rows_per_image: SIZE as u32,
                    };

                    let texture_size = wgpu::Extent3d {
                        width: SIZE as u32,
                        height: SIZE as u32,
                        depth: 1,
                    };

                    let mut encoder = gpu
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                    encoder.copy_buffer_to_texture(
                        wgpu::BufferCopyView {
                            buffer: &game.buffers.render,
                            layout,
                        },
                        wgpu::TextureCopyView {
                            texture: &state.diffuse_texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                        },
                        texture_size,
                    );
                    gpu.queue.submit(Some(encoder.finish()));
                }
            }
            Event::MainEventsCleared => {
                // RedrawRequested will only trigger once, unless we manually
                // request it.
                window.request_redraw();
            }
            _ => {}
        }
    });
}
