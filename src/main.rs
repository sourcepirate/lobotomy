use anyhow::Result;
use clap::Parser;

use lobotomy::model;
use lobotomy::tui::App;

#[derive(Parser, Debug)]
#[command(
    name = "lobotomy",
    about = "TUI to explore and surgically edit layers of HuggingFace safetensors models",
    version
)]
struct Args {
    /// .safetensors path, model directory, or HF repo id (e.g. Qwen/Qwen2.5-0.5B)
    model: String,

    /// Branch/tag/commit for HF repo ids
    #[arg(long)]
    revision: Option<String>,

    /// Only use cached HF files, never hit the network
    #[arg(long)]
    offline: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let index = model::build_index(&args.model, args.revision.as_deref(), args.offline)?;
    let mut app = App::new(index);

    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        ratatui::restore();
        hook(info);
    }));

    let mut terminal = ratatui::init();
    let result = app.run(&mut terminal);
    ratatui::restore();
    result
}
