import { useEffect, useRef, useState } from "react";
import { ChatComposer } from "../components/ChatComposer";
import { ChatSettingsDrawer } from "../components/ChatSettingsDrawer";
import { describeAppError } from "../services/errors";
import { fileToDataUrl } from "../services/media";
import type { ChatMessage, ChatSettings, PageProps } from "../types";

export function ChatPage({ snapshot, service, navigate }: PageProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [images, setImages] = useState<File[]>([]);
  const [generating, setGenerating] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [settings, setSettings] = useState<ChatSettings>({
    systemPrompt: "You are a concise and helpful assistant.",
    temperature: 0.7,
    maxTokens: 1024,
  });
  const bottomRef = useRef<HTMLDivElement>(null);
  const generationRef = useRef(0);
  const runtimeMissing = snapshot.runtime.status !== "ready";
  const noModels = snapshot.models.length === 0;
  const noActiveModel = snapshot.cluster.status !== "running";
  const disabledReason = runtimeMissing
    ? "The inference runtime is not installed. Complete runtime setup before starting chat."
    : noModels
      ? "No model is available. Add a model folder and launch a model first."
      : noActiveModel
        ? "No model is loaded. Choose and launch a model before chatting."
        : "";

  useEffect(
    () => bottomRef.current?.scrollIntoView({ behavior: "smooth" }),
    [messages, generating],
  );

  async function submit(content = draft) {
    if (!content.trim() || disabledReason || generating) return;
    const generationId = ++generationRef.current;
    const attachedImages = images;
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: content.trim(),
      imageNames: images.map((image) => image.name),
    };
    const nextMessages = [...messages, userMessage];
    setMessages(nextMessages);
    setDraft("");
    setImages([]);
    setGenerating(true);
    try {
      const imageDataUrls = await Promise.all(attachedImages.map(fileToDataUrl));
      const response = await service.sendChatMessage(nextMessages, settings, imageDataUrls);
      if (generationRef.current !== generationId) return;
      setMessages([
        ...nextMessages,
        { id: crypto.randomUUID(), role: "assistant", content: response.content },
      ]);
    } catch (reason) {
      if (generationRef.current !== generationId) return;
      setMessages([
        ...nextMessages,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: describeAppError(reason, "Generation stopped unexpectedly."),
          error: true,
        },
      ]);
    } finally {
      if (generationRef.current === generationId) setGenerating(false);
    }
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
  const lastUser = [...messages].reverse().find((message) => message.role === "user");

  return (
    <div className="page chat-page">
      <header className="page-header chat-header">
        <div>
          <p className="section-kicker">Inference session</p>
          <h1>Cluster chat</h1>
        </div>
        <div>
          <span
            className={`status-pill ${snapshot.cluster.status === "running" ? "online" : "offline"}`}
          >
            <i aria-hidden="true" />
            {snapshot.cluster.status === "running" ? "Model ready" : "No model"}
          </span>
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
                  <button
                    className="text-button"
                    disabled={!lastUser}
                    onClick={() => void submit(lastUser?.content)}
                  >
                    Retry
                  </button>
                )}
              </header>
              {message.imageNames?.map((name) => (
                <span className="attachment-chip" key={name}>
                  ▧ {name}
                </span>
              ))}
              <p>{message.content}</p>
            </article>
          ))}
          {generating && (
            <article className="message message-assistant streaming" aria-live="polite">
              <header>
                <span>CLUSTER</span>
                <small>Generating</small>
              </header>
              <p>
                <i aria-hidden="true" />
                <i aria-hidden="true" />
                <i aria-hidden="true" />
              </p>
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
        submit={() => void submit()}
        stop={() => void stop()}
      />
    </div>
  );
}
