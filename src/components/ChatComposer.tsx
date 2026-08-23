import type { FormEvent, KeyboardEvent } from "react";
import {
  ActionIcon,
  Box,
  Button,
  Flex,
  Group,
  Pill,
  Text,
  Textarea,
  VisuallyHidden,
} from "@mantine/core";
import { IconPaperclip } from "@tabler/icons-react";

interface ChatComposerProps {
  draft: string;
  setDraft: (draft: string) => void;
  images: File[];
  setImages: (images: File[]) => void;
  disabledReason: string;
  generating: boolean;
  allowImages: boolean;
  submit: () => void;
  stop: () => void;
}

export function ChatComposer({
  draft,
  setDraft,
  images,
  setImages,
  disabledReason,
  generating,
  allowImages,
  submit,
  stop,
}: ChatComposerProps) {
  function onSubmit(event: FormEvent) {
    event.preventDefault();
    submit();
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  }

  return (
    <Box
      component="form"
      data-testid="chat-composer"
      onSubmit={onSubmit}
      mt="md"
      p="md"
      style={{
        borderRadius: "var(--mantine-radius-xs)",
        border: "1px solid var(--mantine-color-dark-4)",
        background: "var(--mantine-color-dark-8)",
      }}
    >
      {allowImages && (
        <Text size="xs" c="dimmed" mb={4}>
          Image attach is experimental and only sent to vision models.
        </Text>
      )}
      {images.length > 0 && (
        <Flex gap="xs" wrap="wrap" mb="sm">
          {images.map((image) => (
            <Pill
              key={image.name}
              withRemoveButton
              removeButtonProps={{
                "aria-label": `Remove ${image.name}`,
                onClick: () => setImages(images.filter((item) => item !== image)),
              }}
            >
              {image.name}
            </Pill>
          ))}
        </Flex>
      )}
      <Textarea
        aria-label="Message"
        placeholder={disabledReason || "Ask the loaded model…"}
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        onKeyDown={onKeyDown}
        disabled={!!disabledReason}
        autosize
        minRows={2}
        maxRows={8}
        variant="unstyled"
        mb="sm"
      />
      <Group justify="space-between" wrap="nowrap">
        <Group gap="sm" wrap="nowrap">
          {allowImages && (
            <>
              <ActionIcon
                variant="default"
                size="lg"
                aria-label="Choose image files"
                disabled={!!disabledReason}
                onClick={() => document.getElementById("chat-image-input")?.click()}
              >
                <IconPaperclip size={16} />
              </ActionIcon>
              <VisuallyHidden>
                <input
                  id="chat-image-input"
                  aria-label="Attach image"
                  type="file"
                  accept="image/png,image/jpeg,image/webp"
                  multiple
                  disabled={!!disabledReason || !allowImages}
                  onChange={(event) => {
                    setImages([...images, ...Array.from(event.target.files ?? [])]);
                    event.target.value = "";
                  }}
                />
              </VisuallyHidden>
            </>
          )}
          <Text size="xs" c="dimmed">
            {images.length
              ? `${images.length} image${images.length === 1 ? "" : "s"}`
              : "Enter to send · Shift+Enter for new line"}
          </Text>
        </Group>
        {generating ? (
          <Button color="coral" type="button" onClick={stop}>
            ■ Stop
          </Button>
        ) : (
          <Button
            type="submit"
            disabled={!!disabledReason || !draft.trim()}
            aria-label="Send message"
          >
            Send ↗
          </Button>
        )}
      </Group>
    </Box>
  );
}
