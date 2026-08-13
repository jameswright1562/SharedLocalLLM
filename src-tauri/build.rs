use std::{env, fs::File, path::PathBuf};

fn main() {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set"));
    let icon_path = out_dir.join("shared-local-llm.ico");
    let mut rgba = vec![0x10; 32 * 32 * 4];
    for pixel in rgba.chunks_exact_mut(4) {
        pixel.copy_from_slice(&[0x10, 0x1b, 0x2d, 0xff]);
    }
    let image = ico::IconImage::from_rgba_data(32, 32, rgba);
    let mut icon = ico::IconDir::new(ico::ResourceType::Icon);
    icon.add_entry(ico::IconDirEntry::encode(&image).expect("encode icon"));
    icon.write(File::create(&icon_path).expect("create icon"))
        .expect("write icon");

    tauri_build::try_build(
        tauri_build::Attributes::new()
            .windows_attributes(tauri_build::WindowsAttributes::new().window_icon_path(icon_path)),
    )
    .expect("Tauri build failed")
}
