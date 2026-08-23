import type { ChatMessage, ChatSettings } from "../types";

const STORAGE_KEY = "sharedlocalllm.chat.v1";
const MAX_PERSISTED_MESSAGES = 200;

export interface StoredChat {
  messages: ChatMessage[];
  settings: ChatSettings;
}

export function defaultChatSettings(): ChatSettings {
  return {
    systemPrompt: "You are a concise and helpful assistant.",
    temperature: 0.7,
    maxTokens: 1024,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function optionalString(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function parseMessage(value: unknown): ChatMessage | null {
  if (!isRecord(value)) return null;
  const { id, role, content } = value;
  if (typeof id !== "string" || id.length === 0) return null;
  if (role !== "user" && role !== "assistant") return null;
  if (typeof content !== "string") return null;
  if (value.error === true) return null;
  const imageNames = Array.isArray(value.imageNames)
    ? value.imageNames.filter((name): name is string => typeof name === "string")
    : undefined;
  const imageData = Array.isArray(value.imageData)
    ? value.imageData.filter((data): data is string => typeof data === "string")
    : undefined;
  return {
    id,
    role,
    content,
    reasoning: optionalString(value.reasoning),
    tokensPerSecond:
      typeof value.tokensPerSecond === "number" && Number.isFinite(value.tokensPerSecond)
        ? value.tokensPerSecond
        : undefined,
    ...(imageNames !== undefined && imageNames.length > 0 ? { imageNames } : {}),
    ...(imageData !== undefined && imageData.length > 0 ? { imageData } : {}),
  };
}

function parseMessages(value: unknown): ChatMessage[] {
  if (!Array.isArray(value)) return [];
  return value
    .map(parseMessage)
    .filter((message): message is ChatMessage => message !== null)
    .slice(-MAX_PERSISTED_MESSAGES);
}

function parseSettings(value: unknown): ChatSettings {
  const defaults = defaultChatSettings();
  if (!isRecord(value)) return defaults;
  const temperature =
    typeof value.temperature === "number" &&
    Number.isFinite(value.temperature) &&
    value.temperature >= 0 &&
    value.temperature <= 2
      ? value.temperature
      : defaults.temperature;
  const maxTokens =
    typeof value.maxTokens === "number" &&
    Number.isInteger(value.maxTokens) &&
    value.maxTokens >= 64 &&
    value.maxTokens <= 8192
      ? value.maxTokens
      : defaults.maxTokens;
  return {
    systemPrompt:
      typeof value.systemPrompt === "string" ? value.systemPrompt : defaults.systemPrompt,
    temperature,
    maxTokens,
  };
}

function readStorage(): StoredChat {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) throw new Error("empty");
    const parsed: unknown = JSON.parse(raw);
    return {
      messages: parseMessages(isRecord(parsed) ? parsed.messages : undefined),
      settings: parseSettings(isRecord(parsed) ? parsed.settings : undefined),
    };
  } catch {
    return { messages: [], settings: defaultChatSettings() };
  }
}

export function loadStoredChat(): StoredChat {
  if (typeof window === "undefined") return { messages: [], settings: defaultChatSettings() };
  return readStorage();
}

export function saveStoredChat(messages: ChatMessage[], settings: ChatSettings): void {
  const persisted = parseMessages(messages).map((message) =>
    message.role === "user" ? message : { ...message, imageNames: undefined, imageData: undefined },
  );
  const payload = JSON.stringify({ messages: persisted, settings });
  try {
    window.localStorage.setItem(STORAGE_KEY, payload);
  } catch {
    const withoutImages = persisted.map((message) => ({
      ...message,
      imageNames: undefined,
      imageData: undefined,
    }));
    try {
      window.localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({ messages: withoutImages, settings }),
      );
    } catch {
      // Persistence is best effort; chatting continues in memory.
    }
  }
}

export function clearStoredChat(): void {
  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    // Nothing to clean up when storage is unavailable.
  }
}
