import ReactMarkdown from "react-markdown";

function MessageBubble({
    role,
    content,
    intent,
}) {
    const isUser = role === "user";
    const isError = role === "error";

    return (
        <div className={`message-row ${role}`}>
            <div className="message-avatar">
                {isUser ? "You" : isError ? "!" : "A&R"}
            </div>

            <div className="message-content">
                <div className="message-label">
                    {isUser
                        ? "You"
                        : isError
                            ? "Error"
                            : intent === "order"
                                ? "Aster & Row Orders"
                                : "Aster & Row Support"}
                </div>

                <div
                    className={`message-bubble ${intent === "order"
                            ? "order-message"
                            : ""
                        }`}
                >
                    {isUser || isError ? (
                        content
                    ) : (
                        <ReactMarkdown>
                            {content}
                        </ReactMarkdown>
                    )}
                </div>
            </div>
        </div>
    );
}

export default MessageBubble;