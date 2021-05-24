use druid::widget::{Flex, Label};
use druid::{AppLauncher, Widget, WidgetExt, WindowDesc};

#[repr(C)]
#[derive(Copy, Clone, zerocopy::AsBytes, druid::Lens, druid::Data, Debug)]
pub struct Settings {
    pub speed: f32,
    pub sensor_angle: f32,
    pub sensor_distance: f32,
    pub sensor_size: u32,
    pub rotation_speed: f32,
    pub decay_ratio: f32,
    pub diffusion_ratio: f32,
    pub entropy: f32,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            speed: 1.,
            sensor_angle: 1.0,
            sensor_distance: 25.,
            sensor_size: 1,
            rotation_speed: 0.1,
            decay_ratio: 0.01,
            diffusion_ratio: 0.1,
            entropy: 0.1,
        }
    }
}

fn slider(description: &'static str, min: f64, max: f64) -> impl Widget<f32> {
    Flex::row()
        .with_child(
            Label::new(move |data: &f64, _env: &_| format!("{}: {:.2}", description, data))
                .padding(5.0)
                .center(),
        )
        .with_flex_child(
            druid::widget::Slider::new()
                .with_range(min, max)
                .expand_width(),
            1.0,
        )
        .lens(druid::lens::Map::new(
            |x| (*x as f64),
            |x, v| (*x = v as f32),
        ))
}

fn slider_int(description: &'static str, min: u32, max: u32) -> impl Widget<u32> {
    Flex::row()
        .with_child(
            Label::new(move |data: &f64, _env: &_| format!("{}: {}", description, data.round()))
                .padding(5.0)
                .center(),
        )
        .with_flex_child(
            druid::widget::Slider::new()
                .with_range(min as f64, max as f64)
                .expand_width(),
            1.0,
        )
        .lens(druid::lens::Map::new(
            |x| (*x as f64),
            |x, v| (*x = v.round() as u32),
        ))
}

fn ui_builder() -> impl Widget<Settings> {
    // The label text will be computed dynamically based on the current locale and count
    let label = Label::new("Settings").padding(5.0).center();

    Flex::column()
        .with_child(label)
        .with_child(slider("Speed", 0., 3.).lens(Settings::speed))
        .with_child(slider("Sensor angle", 0., 3.14 / 2.).lens(Settings::sensor_angle))
        .with_child(slider("Sensor distance", 0., 15.).lens(Settings::sensor_distance))
        .with_child(slider_int("Sensor size", 0, 3).lens(Settings::sensor_size))
        .with_child(slider("Rotation speed", 0., 3.).lens(Settings::rotation_speed))
        .with_child(slider("Decay", 0., 1.).lens(Settings::decay_ratio))
        .with_child(slider("Diffusion", 0., 1.).lens(Settings::diffusion_ratio))
        .with_child(slider("Entropy", 0., 2.).lens(Settings::entropy))
}

pub struct SettingsManager<CB> {
    callback: CB,
}

impl<CB> SettingsManager<CB>
where
    CB: FnMut(Settings) + Send + 'static,
{
    pub fn new(callback: CB, initial_value: Settings) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            let main_window = WindowDesc::new(ui_builder)
                .title("Settings")
                .show_titlebar(true)
                .resizable(true)
                .set_window_state(druid::WindowState::RESTORED);
            AppLauncher::with_window(main_window)
                .delegate(SettingsManager { callback })
                .launch(initial_value)
                .unwrap();
        })
    }
}

impl<CB> druid::AppDelegate<Settings> for SettingsManager<CB>
where
    CB: FnMut(Settings),
{
    fn event(
        &mut self,
        _ctx: &mut druid::DelegateCtx,
        _window_id: druid::WindowId,
        event: druid::Event,
        data: &mut Settings,
        _env: &druid::Env,
    ) -> Option<druid::Event> {
        (self.callback)(*data);
        Some(event)
    }
}
