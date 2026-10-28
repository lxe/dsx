//! Deterministic stand-in engine for tests and client bring-up: no weights, no GPU.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, anyhow};
use tokenizers::Tokenizer;

use super::engine::{Engine, Finish, GenParams};

pub const MOCK_REASONING: &str = "The user wants a short answer.";
pub const MOCK_REPLY: &str = "Hello from the dsx mock engine.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MockMode {
    /// Reads the decoded prompt and answers like the model would: reasoning when the prompt opens `<think>`,
    /// a DSML call to the first declared tool, a summary after a tool result, else a canned reply.
    Scripted,
    /// Repeats the prompt tokens.
    Echo,
}

pub struct MockEngine {
    tok: Arc<Tokenizer>,
    pub mode: MockMode,
    /// sleep per token (cancellation tests, eyeballing streams)
    pub delay: Duration,
}

impl MockEngine {
    pub fn new(tok: Arc<Tokenizer>, mode: MockMode) -> Self {
        Self { tok, mode, delay: Duration::ZERO }
    }

    /// The reply text the scripted mock produces for a rendered prompt (special tokens included).
    pub fn scripted_reply(prompt: &str) -> String {
        const EOS: &str = "<｜end▁of▁sentence｜>";
        const DSML: &str = "｜DSML｜";
        // inside a forced tool-call block (tool_choice required): the prompt already opened `<｜DSML｜ calls>`
        let forced = prompt.ends_with(&format!("<{DSML} calls>\n"));
        let thinking = prompt.ends_with("<think>");
        // the turn being answered: everything after the last finished assistant turn
        let last = prompt.rsplit(EOS).next().unwrap_or(prompt);
        let tool = first_tool(prompt);
        let mut out = String::new();
        if thinking {
            out += MOCK_REASONING;
            out += "</think>";
        }
        if let Some(i) = last.rfind("<tool_result>") {
            let res = &last[i + "<tool_result>".len()..];
            let res = res.split("</tool_result>").next().unwrap_or(res);
            out += &format!("The tool returned: {}", res.chars().take(200).collect::<String>());
        } else if let Some((name, required)) = tool {
            let params: Vec<String> = required
                .iter()
                .map(|p| format!("<{DSML} parameter name=\"{p}\" string=\"true\">mock</{DSML} parameter>"))
                .collect();
            let invoke = format!("<{DSML} invoke name=\"{name}\">\n{}\n</{DSML} invoke>\n</{DSML} calls>", params.join("\n"));
            if forced {
                out = invoke; // no reasoning inside a forced block
            } else {
                out += &format!("I'll call {name}.\n\n<{DSML} calls>\n{invoke}");
            }
        } else {
            out += MOCK_REPLY;
        }
        out += EOS;
        out
    }
}

/// First tool in the rendered `### Available Tool Schemas` section (one JSON object per line): name and `required`.
fn first_tool(prompt: &str) -> Option<(String, Vec<String>)> {
    let s = prompt.split("### Available Tool Schemas").nth(1)?;
    let v: serde_json::Value = serde_json::from_str(s.trim_start().lines().next()?).ok()?;
    let name = v["name"].as_str()?.to_string();
    let required = v["parameters"]["required"].as_array().map(|r| r.iter().filter_map(|p| p.as_str().map(String::from)).collect());
    Some((name, required.unwrap_or_default()))
}

impl Engine for MockEngine {
    fn generate(&mut self, prompt: &[u32], _p: &GenParams, emit: &mut dyn FnMut(u32) -> bool) -> Result<Finish> {
        let reply: Vec<u32> = match self.mode {
            MockMode::Echo => prompt.to_vec(),
            MockMode::Scripted => {
                let text = self.tok.decode(prompt, false).map_err(|e| anyhow!("{e}"))?;
                let reply = Self::scripted_reply(&text);
                self.tok.encode(reply, false).map_err(|e| anyhow!("{e}"))?.get_ids().to_vec()
            }
        };
        for t in reply {
            if !self.delay.is_zero() {
                std::thread::sleep(self.delay);
            }
            if !emit(t) {
                return Ok(Finish::Cancelled);
            }
        }
        Ok(Finish::Stop)
    }
}
