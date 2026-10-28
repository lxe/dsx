//! Render an OpenAI chat-completions request (JSON file) to the V4.1 prompt text exactly as the server's `encode`
//! does (minus tokenization): `cargo run -p dsx-core --example render_chat -- req.json`
use deepseek_recipe::openai::ChatCompletionRequest;
use deepseek_recipe::request::{ConversionOptions, ProtocolRequest};
use deepseek_recipe_encoding::PromptEncoding;
use deepseek_recipe_encoding::v4::dsv41::DeepseekV41Encoding;

fn main() {
    let p = std::env::args().nth(1).expect("request JSON path");
    let req: ChatCompletionRequest = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
    let conv = req.convert(ConversionOptions::default()).unwrap();
    print!("{}", DeepseekV41Encoding::new().render_conversation(&conv.conversation).prompt);
}
