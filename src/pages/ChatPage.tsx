import { useEffect, useRef, useState } from "react";
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
    <div className="page chat-page">
      <header className="page-header chat-header">
        <div>
          <p className="section-kicker">Inference session</p>
          <h1>Cluster chat</h1>
        </div>
        <div>
          <span className={`status-pill ${!noActiveModel ? "online" : "offline"}`}>
            <i aria-hidden="true" />
            {!noActiveModel ? "Model ready" : "No model"}
          </span>
          {(localRunning || peerRunning) && (
            <button
              className="button stop-button compact-button"
              onClick={() => void service.stopCluster().then(() => refreshSnapshot())}
            >
              Stop cluster
            </button>
          )}
          {messages.length > 0 && (
            <button className="button secondary compact-button" onClick={clearChat}>
              Clear chat
            </button>
          )}
          <button
            className="button secondary compact-button"
            onClick={() => setDrawerOpen(!drawerOpen)}
          >
            Generation settings
          </button>
        </div>
      </header>
      {disabledReason && (
        <div className="error-panel chat-error" role="alert">
          <span className="status-dot warning" aria-hidden="true" />
          <div>
            <strong>Chat unavailable</strong>
            <p>{disabledReason}</p>
            <button
              className="text-button"
              onClick={() => navigate(runtimeMissing ? "settings" : "models")}
            >
              {runtimeMissing ? "Open runtime settings" : "Choose a model"} →
            </button>
          </div>
        </div>
      )}
      {peerRunning && !localRunning && (
        <p className="inline-success">
          Chat is proxied through the computer that launched the model.
        </p>
      )}
      <div className="chat-workspace">
        <section className="message-stage" aria-label="Conversation">
          {messages.length === 0 && !disabledReason && (
            <div className="chat-empty">
              <span className="chat-orbit" aria-hidden="true">
                <i />
                <b>LLM</b>
              </span>
              <h2>Send work through the cluster</h2>
              <p>
                Messages stay local. The coordinator routes model layers over the authenticated peer
                channel.
              </p>
            </div>
          )}
          {messages.map((message) => (
            <article
              className={`message message-${message.role} ${message.error ? "message-error" : ""}`}
              key={message.id}
            >
              <header>
                <span>{message.role === "user" ? "YOU" : "CLUSTER"}</span>
                {message.error && (
                  <button className="text-button" onClick={() => void retry()}>
                    Retry
                  </button>
                )}
              </header>
              {message.imageNames?.map((name) => (
                <span className="attachment-chip" key={name}>
                  ▧ {name}
                </span>
              ))}
              {message.reasoning && (
                <details className="message-reasoning">
                  <summary>Reasoning</summary>
                  <p>{message.reasoning}</p>
                </details>
              )}
              <p>{message.content}</p>
              {message.tokensPerSecond !== undefined && (
                <p className="message-stats">{message.tokensPerSecond} tok/s</p>
              )}
            </article>
          ))}
          {generating && (
            <article className="message message-assistant streaming" aria-live="polite">
              <header>
                <span>CLUSTER</span>
                <small>{phase === "generating" ? "Generating" : "Processing prompt"}</small>
              </header>
              {streamingReasoning && (
                <details className="message-reasoning" open>
                  <summary>Reasoning</summary>
                  <p className="streaming-text">{streamingReasoning}</p>
                </details>
              )}
              {streaming ? (
                <p className="streaming-text">{streaming}</p>
              ) : streamingReasoning ? null : (
                <p>
                  <i aria-hidden="true" />
                  <i aria-hidden="true" />
                  <i aria-hidden="true" />
                </p>
              )}
            </article>
          )}
          <div ref={bottomRef} />
        </section>
        {drawerOpen && (
          <ChatSettingsDrawer
            settings={settings}
            setSettings={setSettings}
            close={() => setDrawerOpen(false)}
          />
        )}
      </div>
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
    </div>
  );
}
