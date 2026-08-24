import { useEffect, useRef, useState } from "react";
import { Badge, Box, Button, Flex, Group, Paper, Pill, Stack, Text, Title } from "@mantine/core";

import { ChatComposer } from "../components/ChatComposer";
import { ChatSettingsDrawer } from "../components/ChatSettingsDrawer";
import { clearStoredChat, loadStoredChat, saveStoredChat } from "../services/chatStorage";
import { describeAppError } from "../services/errors";
import { fileToDataUrl } from "../services/media";
import type { ChatMessage, ChatSettings, PageProps } from "../types";

export function ChatPage({ snapshot, service, navigate, refreshSnapshot }: PageProps) {
  const [stored] = useState(loadStoredChat);
  const [messages, setMessages] = useState<ChatMessage[]>(stored.messages);
  const [draft, setDraft] = useState("");
  const [images, setImages] = useState<File[]>([]);
  const [generating, setGenerating] = useState(false);
  const [streaming, setStreaming] = useState("");
  const [streamingReasoning, setStreamingReasoning] = useState("");
  const [phase, setPhase] = useState<"processing" | "generating">("processing");
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [settings, setSettings] = useState<ChatSettings>(stored.settings);
  const bottomRef = useRef<HTMLDivElement>(null);
  const generationRef = useRef(0);
  const streamedTps = useRef<number | undefined>(undefined);
  const localRunning = snapshot.cluster.status === "running";
  const peerRunning = snapshot.nodes.some(
    (node, index) => index > 0 && node.clusterStatus === "running",
  );
  const activeModel =
    snapshot.models.find((model) => model.id === snapshot.cluster.modelId) ??
    snapshot.models.find(
      (model) => model.id === snapshot.nodes.find((node) => node.clusterModelId)?.clusterModelId,
    );
  const visionEnabled = activeModel?.capability === "vision";
  const runtimeMissing = snapshot.runtime.status !== "ready";
  const noModels = snapshot.models.length === 0;
  const noActiveModel = !localRunning && !peerRunning;
  const disabledReason = runtimeMissing
    ? "The inference runtime is not installed. Complete runtime setup before starting chat."
    : noModels
      ? "No model is available. Add a model folder and launch a model first."
      : noActiveModel
        ? "No model is loaded. Choose and launch a model before chatting."
        : "";

  useEffect(() => {
    saveStoredChat(messages, settings);
  }, [messages, settings]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, generating, streaming]);

  function clearChat() {
    clearStoredChat();
    setMessages([]);
  }

  async function submit(content = draft, existing?: ChatMessage[], retryImages: string[] = []) {
    if (!content.trim() || disabledReason || generating) return;
    const generationId = ++generationRef.current;
    const attachedImages = images;
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: content.trim(),
      imageNames: attachedImages.map((image) => image.name),
    };
    const history = existing ?? [...messages.filter((message) => !message.error), userMessage];
    if (!existing) {
      setMessages(history);
      setDraft("");
      setImages([]);
    }
    setGenerating(true);
    setStreaming("");
    setStreamingReasoning("");
    streamedTps.current = undefined;
    setPhase("processing");
    const assistantId = crypto.randomUUID();
    try {
      const imageDataUrls =
        retryImages.length > 0 ? retryImages : await Promise.all(attachedImages.map(fileToDataUrl));
      if (!existing) history[history.length - 1] = { ...userMessage, imageData: imageDataUrls };
      const wire = history.map((message) => ({
        id: message.id,
        role: message.role,
        content: message.content,
        imageNames: message.imageNames,
      }));
      const response = await service.sendChatMessage(
        wire.filter((message) => message.role !== "system"),
        settings,
        visionEnabled ? imageDataUrls : [],
        (event) => {
          if (generationRef.current !== generationId) return;
          if (event.kind === "status") {
            setPhase("processing");
          } else if (event.kind === "reasoning") {
            setPhase("generating");
            setStreamingReasoning((current) => current + event.content);
          } else if (event.kind === "token") {
            setPhase("generating");
            setStreaming((current) => current + event.content);
          } else if (event.kind === "stats") {
            streamedTps.current = event.tokensPerSecond;
          }
        },
      );
      if (generationRef.current !== generationId) return;
      setMessages([
        ...history,
        {
          id: assistantId,
          role: "assistant",
          content: response.content,
          reasoning: response.reasoning,
          tokensPerSecond: response.tokensPerSecond ?? streamedTps.current,
        },
      ]);
    } catch (reason) {
      if (generationRef.current !== generationId) return;
      setMessages([
        ...history,
        {
          id: assistantId,
          role: "assistant",
          content: describeAppError(reason, "Generation stopped unexpectedly."),
          error: true,
        },
      ]);
    } finally {
      if (generationRef.current === generationId) {
        setGenerating(false);
        setStreaming("");
      }
    }
  }

  async function retry() {
    const lastUser = [...messages].reverse().find((message) => message.role === "user");
    if (!lastUser) return;
    const cutoff = messages.findIndex((message) => message.id === lastUser.id);
    const history = messages.slice(0, cutoff + 1).filter((message) => !message.error);
    setMessages(history);
    await submit(lastUser.content, history, lastUser.imageData ?? []);
  }

  async function stop() {
    generationRef.current += 1;
    setGenerating(false);
    try {
      await service.cancelGeneration();
    } catch (reason) {
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: describeAppError(reason, "Generation cancellation failed."),
          error: true,
        },
      ]);
    }
  }

  return (
    <Box mih="100%" display="flex" style={{ flexDirection: "column" }}>
      <Flex justify="space-between" align="flex-start" gap="md" wrap="wrap" mb="md">
        <Box>
          <Text size="xs" fw={700} tt="uppercase" lts={1.5} c="cyan">
            Inference session
          </Text>
          <Title order={1}>Cluster chat</Title>
        </Box>
        <Group gap="xs">
          <Badge color={!noActiveModel ? "mint" : "gray"} variant="light">
            {!noActiveModel ? "Model ready" : "No model"}
          </Badge>
          {(localRunning || peerRunning) && (
            <Button
              color="coral"
              variant="light"
              size="compact-sm"
              onClick={() => void service.stopCluster().then(() => refreshSnapshot())}
            >
              Stop cluster
            </Button>
          )}
          {messages.length > 0 && (
            <Button variant="default" size="compact-sm" onClick={clearChat}>
              Clear chat
            </Button>
          )}
          <Button variant="default" size="compact-sm" onClick={() => setDrawerOpen(!drawerOpen)}>
            Generation settings
          </Button>
        </Group>
      </Flex>

      {disabledReason && (
        <Paper
          role="alert"
          withBorder
          p="md"
          mb="md"
          bg="dark.8"
          style={{ borderColor: "var(--mantine-color-amber-5)" }}
        >
          <Group gap="sm" align="flex-start" wrap="nowrap">
            <span
              aria-hidden="true"
              style={{
                display: "inline-block",
                width: 8,
                height: 8,
                borderRadius: "50%",
                marginTop: 6,
                flex: "0 0 auto",
                background: "var(--mantine-color-amber-4)",
              }}
            />
            <Stack gap={4}>
              <Text fw={600}>Chat unavailable</Text>
              <Text size="sm" c="dimmed">
                {disabledReason}
              </Text>
              <Button
                variant="subtle"
                size="compact-sm"
                onClick={() => navigate(runtimeMissing ? "settings" : "models")}
              >
                {runtimeMissing ? "Open runtime settings" : "Choose a model"} →
              </Button>
            </Stack>
          </Group>
        </Paper>
      )}
      {peerRunning && !localRunning && (
        <Text size="sm" c="mint" mb="md">
          Chat is proxied through the computer that launched the model.
        </Text>
      )}

      <section aria-label="Conversation" style={{ flex: 1 }}>
        <Stack gap="md" maw={860}>
          {messages.length === 0 && !disabledReason && (
            <Stack align="center" gap="xs" ta="center" py="xl">
              <Box pos="relative" className="chat-orbit" aria-hidden="true">
                <i />
                <b>LLM</b>
              </Box>
              <Title order={3} mt="xs">
                Send work through the cluster
              </Title>
              <Text c="dimmed" maw={420}>
                Messages stay local. The coordinator routes model layers over the authenticated peer
                channel.
              </Text>
            </Stack>
          )}
          {messages.map((message) => (
            <MessageBubble key={message.id} message={message} onRetry={() => void retry()} />
          ))}
          {generating && (
            <Paper p="md" bg="dark.7" style={{ alignSelf: "stretch" }} aria-live="polite">
              <Group justify="space-between" mb={4}>
                <Text size="10px" tt="uppercase" lts={2} c="cyan" fw={600}>
                  Cluster
                </Text>
                <Text size="10px" c="dimmed" tt="uppercase">
                  {phase === "generating" ? "Generating" : "Processing prompt"}
                </Text>
              </Group>
              {streamingReasoning && (
                <details open>
                  <summary>
                    <Text size="xs" c="dimmed" span>
                      Reasoning
                    </Text>
                  </summary>
                  <Text size="sm" c="dimmed">
                    {streamingReasoning}
                  </Text>
                </details>
              )}
              {streaming ? (
                <Text>{streaming}</Text>
              ) : streamingReasoning ? null : (
                <span className="streaming-dots" aria-hidden="true">
                  <i />
                  <i />
                  <i />
                </span>
              )}
            </Paper>
          )}
          <div ref={bottomRef} />
        </Stack>
      </section>

      <ChatComposer
        draft={draft}
        setDraft={setDraft}
        images={images}
        setImages={setImages}
        disabledReason={disabledReason}
        generating={generating}
        allowImages={visionEnabled}
        submit={() => void submit()}
        stop={() => void stop()}
      />

      {drawerOpen && (
        <ChatSettingsDrawer
          settings={settings}
          setSettings={setSettings}
          close={() => setDrawerOpen(false)}
        />
      )}
    </Box>
  );
}

function MessageBubble({ message, onRetry }: { message: ChatMessage; onRetry: () => void }) {
  const isUser = message.role === "user";
  return (
    <Paper
      p="md"
      bg={isUser ? "dark.6" : "dark.7"}
      withBorder
      style={{
        alignSelf: isUser ? "flex-end" : "stretch",
        maxWidth: isUser ? "85%" : undefined,
        borderColor: message.error ? "var(--mantine-color-coral-5)" : undefined,
      }}
    >
      <Group justify="space-between" mb={4}>
        <Text size="10px" tt="uppercase" lts={2} fw={600} c={message.error ? "coral" : "cyan"}>
          {isUser ? "You" : "Cluster"}
        </Text>
        {message.error && (
          <Button variant="subtle" size="compact-xs" onClick={onRetry}>
            Retry
          </Button>
        )}
      </Group>
      {message.imageNames?.map((name) => (
        <Pill key={name} size="sm" mr={4}>
          ▧ {name}
        </Pill>
      ))}
      {message.reasoning && (
        <details>
          <summary>
            <Text size="xs" c="dimmed" span>
              Reasoning
            </Text>
          </summary>
          <Text size="sm" c="dimmed" mb="xs">
            {message.reasoning}
          </Text>
        </details>
      )}
      <Text size="sm">{message.content}</Text>
      {message.tokensPerSecond !== undefined && (
        <Text size="10px" c="dimmed" mt={4}>
          {message.tokensPerSecond} tok/s
        </Text>
      )}
    </Paper>
  );
}
