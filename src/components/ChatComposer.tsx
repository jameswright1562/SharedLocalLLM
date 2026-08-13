import type { FormEvent, KeyboardEvent } from "react";

interface ChatComposerProps {
  draft: string;
  setDraft: (draft: string) => void;
  images: File[];
  setImages: (images: File[]) => void;
  disabledReason: string;
  generating: boolean;
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
    <form className="chat-composer" data-testid="chat-composer" onSubmit={onSubmit}>
      {images.length > 0 && (
        <div className="composer-attachments">
          {images.map((image) => (
            <span className="attachment-chip" key={image.name}>
              {image.name}
              <button
                type="button"
                aria-label={`Remove ${image.name}`}
                onClick={() => setImages(images.filter((item) => item !== image))}
              >
                ×
              </button>
            </span>
          ))}
        </div>
      )}
      <textarea
        aria-label="Message"
        placeholder={disabledReason || "Ask the loaded model…"}
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        onKeyDown={onKeyDown}
        disabled={!!disabledReason}
        rows={2}
      />
      <div className="composer-actions">
        <label
          className={`icon-button attach-button ${disabledReason ? "disabled" : ""}`}
          aria-label="Attach image"
        >
          ▧
          <input
            type="file"
            accept="image/png,image/jpeg,image/webp"
            multiple
            disabled={!!disabledReason}
            onChange={(event) => setImages(Array.from(event.target.files ?? []))}
          />
        </label>
        <span>
          {images.length
            ? `${images.length} image${images.length === 1 ? "" : "s"}`
            : "Enter to send · Shift+Enter for new line"}
        </span>
        {generating ? (
          <button type="button" className="button stop-button" onClick={stop}>
            ■ Stop
          </button>
        ) : (
          <button
            type="submit"
            className="button primary"
            disabled={!!disabledReason || !draft.trim()}
            aria-label="Send message"
          >
            Send ↗
          </button>
        )}
      </div>
    </form>
  );
}
