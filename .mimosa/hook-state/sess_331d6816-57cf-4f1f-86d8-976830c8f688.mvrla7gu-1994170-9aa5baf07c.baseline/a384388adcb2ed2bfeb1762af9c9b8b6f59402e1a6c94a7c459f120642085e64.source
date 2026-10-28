# Coding agents

For the [pi](https://github.com/badlogic/pi-mono) coding agent, add this to `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "dsx": {
      "baseUrl": "http://localhost:8090/v1",
      "api": "openai-completions",
      "apiKey": "none",
      "authHeader": false,
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": true,
        "maxTokensField": "max_tokens"
      },
      "models": [
        {
          "id": "deepseek-v4.1-flash",
          "reasoning": true,
          "input": ["text", "image"],
          "contextWindow": 262144,
          "maxTokens": 262144,
          "thinkingLevelMap": {
            "off": "none",
            "minimal": null,
            "low": "low",
            "medium": null,
            "high": "high",
            "xhigh": null,
            "max": "max"
          }
        }
      ]
    }
  }
}
```

```bash
pi --provider dsx --model deepseek-v4.1-flash:off
```

With `--api-key`: drop `"authHeader": false`, set `"apiKey"` to the key and point `baseUrl` at the server's URL.

Set the provider context window to the server’s `--ctx` value. The example uses 256K (`262144`); reduce it when the server runs with a smaller context.

Use `deepseek-v4.1-flash:low` to enable low reasoning. `maxTokens` caps each response, including reasoning tokens; increase it for longer tasks while leaving room for the prompt within the context window.

## Reproducible local gauntlet

The [Pi configuration](../bench/pi/models.json) declares text and image input,
a 256K context, and temperature 1. The [launcher](../bench/pi/launch.sh) creates
an isolated run with reasoning off, a fresh random seed, and no inherited code
or conversation. It refuses an existing run directory. It requires Pi, Python 3,
and a server running with `--ctx 262144` and `--vision /path/to/vision.safetensors`.
Image support in Pi's configuration allows image blocks to reach the server;
the server must also have the vision weights loaded.

```bash
bench/pi/launch.sh /tmp/cube-run /path/to/task.txt http://localhost:8090/v1
```

Keep recordings, generated programs, and session files in the run directory,
outside this repository. The launcher submits the task once; Pi controls its
own preview and completion. A run's effective configuration and seed are saved
in `agent/models.json` and `manifest.json`. Changes to that configuration take
effect when Pi starts; restart Pi after changing image capabilities.
