function ChatInput({
    value,
    onChange,
    onSend,
    loading,
}) {
    function handleKeyDown(event) {
        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {
            event.preventDefault();
            onSend();
        }
    }

    return (
        <footer className="input-area">
            <div className="input-wrapper">
                <textarea
                    value={value}
                    onChange={(event) =>
                        onChange(event.target.value)
                    }
                    onKeyDown={handleKeyDown}
                    placeholder="Ask about returns, orders, shipping..."
                    rows={1}
                    disabled={loading}
                />

                <button
                    onClick={onSend}
                    disabled={
                        !value.trim() || loading
                    }
                >
                    {loading ? "..." : "Send"}
                </button>
            </div>

            <p className="disclaimer">
                Aster & Row Support Agent · AI-generated
                responses are based on approved company
                information.
            </p>
        </footer>
    );
}

export default ChatInput;