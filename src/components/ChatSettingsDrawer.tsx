import {
  CloseButton,
  Drawer,
  Group,
  Slider,
  Stack,
  Text,
  Textarea,
  TextInput,
  Title,
} from "@mantine/core";

import type { ChatSettings } from "../types";

interface ChatSettingsDrawerProps {
  settings: ChatSettings;
  setSettings: (settings: ChatSettings) => void;
  close: () => void;
}

export function ChatSettingsDrawer({ settings, setSettings, close }: ChatSettingsDrawerProps) {
  return (
    <Drawer opened onClose={close} position="right" withCloseButton={false}>
      <Stack gap="md">
        <Group justify="space-between" wrap="nowrap">
          <Title order={3}>Generation settings</Title>
          <CloseButton aria-label="Close generation settings" onClick={close} />
        </Group>
        <Textarea
          id="system-prompt"
          label="System prompt"
          value={settings.systemPrompt}
          onChange={(event) => setSettings({ ...settings, systemPrompt: event.target.value })}
          rows={6}
        />
        <Stack gap={4}>
          <Group justify="space-between">
            <Text size="sm" fw={500}>
              Temperature
            </Text>
            <Text size="sm" ff="monospace" fw={600}>
              {settings.temperature.toFixed(1)}
            </Text>
          </Group>
          <Slider
            aria-label="Temperature"
            min={0}
            max={2}
            step={0.1}
            value={settings.temperature}
            onChange={(value) => setSettings({ ...settings, temperature: value })}
            color="cyan"
            label={(value) => value.toFixed(1)}
          />
        </Stack>
        <TextInput
          id="max-tokens"
          label="Maximum response tokens"
          type="number"
          min={64}
          max={8192}
          w={160}
          value={settings.maxTokens}
          onChange={(event) => setSettings({ ...settings, maxTokens: Number(event.target.value) })}
        />
      </Stack>
    </Drawer>
  );
}
