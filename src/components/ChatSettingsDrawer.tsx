import type { ChatSettings } from "../types";

interface ChatSettingsDrawerProps {
  settings: ChatSettings;
  setSettings: (settings: ChatSettings) => void;
  close: () => void;
}

export function ChatSettingsDrawer({ settings, setSettings, close }: ChatSettingsDrawerProps) {
  return (
    <aside className="chat-settings" aria-label="Generation settings">
      <header>
        <h2>Generation settings</h2>
        <button className="icon-button" aria-label="Close generation settings" onClick={close}>
          ×
        </button>
      </header>
      <label className="field-label" htmlFor="system-prompt">
        System prompt
      </label>
      <textarea
        id="system-prompt"
        value={settings.systemPrompt}
        onChange={(event) => setSettings({ ...settings, systemPrompt: event.target.value })}
        rows={6}
      />
      <label className="range-label" htmlFor="temperature">
        <span>Temperature</span>
        <output>{settings.temperature.toFixed(1)}</output>
      </label>
      <input
        id="temperature"
        type="range"
        min="0"
        max="2"
        step="0.1"
        value={settings.temperature}
        onChange={(event) => setSettings({ ...settings, temperature: Number(event.target.value) })}
      />
      <label className="field-label" htmlFor="max-tokens">
        Maximum response tokens
      </label>
      <input
        id="max-tokens"
        type="number"
        min="64"
        max="8192"
        value={settings.maxTokens}
        onChange={(event) => setSettings({ ...settings, maxTokens: Number(event.target.value) })}
      />
    </aside>
  );
}
