import type { FormEvent, KeyboardEvent } from "react";

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
    <form className="chat-composer" data-testid="chat-composer" onSubmit={onSubmit}>
      {allowImages && (
        <p className="estimate-note">
          Image attach is experimental and only sent to vision models.
        </p>
      )}
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
        {allowImages && (
          <button
            type="button"
            className={`icon-button attach-button ${disabledReason ? "disabled" : ""}`}
            aria-label="Choose image files"
            disabled={!!disabledReason}
            onClick={() => document.getElementById("chat-image-input")?.click()}
          >
            ▧
          </button>
        )}
        <input
          id="chat-image-input"
          className="visually-hidden"
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
