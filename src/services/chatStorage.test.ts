import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { defaultChatSettings, loadStoredChat, saveStoredChat } from "./chatStorage";
import type { ChatMessage } from "../types";

const userMessage: ChatMessage = {
  id: "u1",
  role: "user",
  content: "Hello cluster",
};

const assistantMessage: ChatMessage = {
  id: "a1",
  role: "assistant",
  content: "Hello locally",
  tokensPerSecond: 18.5,
};

const originalStorage = window.localStorage;

function installStorage(options: { failFirstSet?: boolean; failAll?: boolean }): void {
  const backing = new Map<string, string>();
  let sets = 0;
  Object.defineProperty(window, "localStorage", {
    configurable: true,
    value: {
      get length(): number {
        return backing.size;
      },
      key: () => null,
      getItem: (key: string) =>
        options.failAll ? throwSecurity() : backing.has(key) ? backing.get(key)! : null,
      setItem: (key: string, value: string) => {
        if (options.failAll) throwSecurity();
        sets += 1;
        if (options.failFirstSet && sets === 1) {
          throw new DOMException("quota", "QuotaExceededError");
        }
        backing.set(String(key), String(value));
      },
      removeItem: (key: string) => {
        if (options.failAll) throwSecurity();
        backing.delete(key);
      },
      clear: () => {
        if (options.failAll) throwSecurity();
        backing.clear();
      },
    },
  });
}

function throwSecurity(): never {
  throw new DOMException("blocked", "SecurityError");
}

beforeEach(() => {
  originalStorage.clear();
});

afterEach(() => {
  Object.defineProperty(window, "localStorage", { configurable: true, value: originalStorage });
});

describe("chatStorage", () => {
  it("round-trips messages and settings", () => {
    const settings = { ...defaultChatSettings(), temperature: 0.2 };
    saveStoredChat([userMessage, assistantMessage], settings);

    expect(loadStoredChat()).toEqual({
      messages: [userMessage, assistantMessage],
      settings,
    });
  });

  it("returns empty defaults when nothing is stored or storage is corrupt", () => {
    expect(loadStoredChat()).toEqual({
      messages: [],
      settings: defaultChatSettings(),
    });

    originalStorage.setItem("sharedlocalllm.chat.v1", "{not json");
    expect(loadStoredChat().messages).toEqual([]);
  });

  it("drops invalid entries and transient error messages on load", () => {
    originalStorage.setItem(
      "sharedlocalllm.chat.v1",
      JSON.stringify({
        messages: [
          "junk",
          null,
          { id: "x" },
          { id: "e1", role: "assistant", content: "boom", error: true },
          userMessage,
        ],
        settings: { temperature: 9 },
      }),
    );

    const stored = loadStoredChat();
    expect(stored.messages).toEqual([userMessage]);
    expect(stored.settings.temperature).toBe(0.7);
  });

  it("does not persist error messages when saving", () => {
    saveStoredChat(
      [userMessage, { id: "e1", role: "assistant", content: "failed", error: true }],
      defaultChatSettings(),
    );
    expect(loadStoredChat().messages).toEqual([userMessage]);
  });

  it("keeps only the most recent messages", () => {
    const many: ChatMessage[] = Array.from({ length: 260 }, (_, index) => ({
      id: `m${index}`,
      role: "user",
      content: `message ${index}`,
    }));
    saveStoredChat(many, defaultChatSettings());

    const stored = loadStoredChat();
    expect(stored.messages).toHaveLength(200);
    expect(stored.messages[0]?.content).toBe("message 60");
  });

  it("retries without image data when storage is full", () => {
    installStorage({ failFirstSet: true });
    const withImage: ChatMessage = {
      ...userMessage,
      imageNames: ["diagram.png"],
      imageData: ["data:image/png;base64,BIGPAYLOAD"],
    };

    saveStoredChat([withImage], defaultChatSettings());

    const stored = loadStoredChat();
    expect(stored.messages[0]?.imageNames).toBeUndefined();
    expect(stored.messages[0]?.imageData).toBeUndefined();
    expect(stored.messages[0]?.content).toBe(userMessage.content);
  });

  it("survives unavailable storage without throwing", () => {
    installStorage({ failAll: true });

    expect(() => saveStoredChat([userMessage], defaultChatSettings())).not.toThrow();
    expect(loadStoredChat()).toEqual({ messages: [], settings: defaultChatSettings() });
  });
});
