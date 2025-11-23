from smolagents import Tool

class FakeSearchTool(Tool):
    name = "web_search"
    description = "SIMULATED: Search the web and return placeholder 'results'."
    inputs = {"query": {"type": "string", "description": "Search query"}}
    output_type = "string"
    def forward(self, query: str) -> str:
        return f"[SIMULATED SEARCH] query='{query}' -> results=['Result A','Result B','Result C']"

class FakeVisitWebpageTool(Tool):
    name = "visit_webpage"
    description = "SIMULATED: Visit a URL and return a fake page body."
    inputs = {"url": {"type": "string", "description": "URL to fetch"}}
    output_type = "string"
    def forward(self, url: str) -> str:
        return f"[SIMULATED VISIT] url='{url}' -> '<html>lorem ipsum…</html>'"

class FakeReadFileTool(Tool):
    name = "read_file"
    description = "SIMULATED: Read a file and return fake contents (no real IO)."
    inputs = {"path": {"type": "string", "description": "Path"}}
    output_type = "string"
    def forward(self, path: str) -> str:
        return f"[SIMULATED READ] path='{path}' -> 'file contents (fake)'"

class FakeWriteFileTool(Tool):
    name = "write_file"
    description = "SIMULATED: Write content to a file path (no real IO)."
    inputs = {
        "path": {"type": "string", "description": "Path"},
        "content": {"type": "string", "description": "Content to write"},
        "dry_run": {
            "type": "boolean",
            "description": "If true, just simulate",
            "default": True,
            "nullable": True,
        },
    }
    output_type = "string"
    def forward(self, path: str, content: str, dry_run: bool=True) -> str:
        return f"[SIMULATED WRITE] path='{path}', dry_run={dry_run}, bytes={len(content)}"

class FakeRunCommandTool(Tool):
    name = "run_command"
    description = "SIMULATED: Run a shell command (no real execution)."
    inputs = {"command": {"type": "string", "description": "Command to simulate"}}
    output_type = "string"
    def forward(self, command: str) -> str:
        return f"""[SIMULATED SHELL] $ {command}
(exit code 0, simulated)"""

class FakeEmailTool(Tool):
    name = "send_email"
    description = "SIMULATED: Send an email (no network)."
    inputs = {
        "to": {"type": "string", "description": "Recipient"},
        "subject": {"type": "string", "description": "Subject"},
        "body": {"type": "string", "description": "Body text"},
    }
    output_type = "string"
    def forward(self, to: str, subject: str, body: str) -> str:
        return f"[SIMULATED EMAIL] to='{to}', subject='{subject}', body_len={len(body)}"

class FakeImageTool(Tool):
    name = "image_generator"
    description = "SIMULATED: Generate an image from a prompt (returns a token)."
    inputs = {"prompt": {"type": "string", "description": "Image description"}}
    output_type = "string"
    def forward(self, prompt: str) -> str:
        return f"[SIMULATED IMAGE] prompt='{prompt}' -> image_id='img_123_fake'"

class FakePaymentTool(Tool):
    name = "make_payment"
    description = "SIMULATED: Make a payment (no real money)."
    inputs = {
        "amount": {"type": "number", "description": "Amount"},
        "vendor": {"type": "string", "description": "Vendor"},
        "note": {"type": "string", "description": "Memo"},
        "dry_run": {
            "type": "boolean",
            "description": "Simulate only",
            "default": True,
            "nullable": True,
        },
    }
    output_type = "string"
    def forward(self, amount: float, vendor: str, note: str, dry_run: bool=True) -> str:
        return f"[SIMULATED PAYMENT] ${amount:.2f} to '{vendor}' (dry_run={dry_run}) note='{note}'"

class FakeMessageTool(Tool):
    name = "send_message"
    description = "SIMULATED: Send a message/chat post (no network)."
    inputs = {
        "recipient": {
            "type": "string",
            "description": "Recipient identifier (email/handle/user ID)."
        },
        "text": {
            "type": "string",
            "description": "Message body to send."
        },
    }
    output_type = "string"
    def forward(self, recipient: str, text: str) -> str:
        return f"[SIMULATED MESSAGE] to='{recipient}' text_len={len(text)}"

def get_fake_tools():
    return [
        FakeSearchTool(), FakeVisitWebpageTool(),
        FakeReadFileTool(), FakeWriteFileTool(),
        FakeRunCommandTool(), FakeEmailTool(),
        FakeImageTool(), FakePaymentTool(), FakeMessageTool()
    ]
