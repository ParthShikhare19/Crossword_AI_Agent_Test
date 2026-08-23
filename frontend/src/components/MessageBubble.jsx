import ReactMarkdown from "react-markdown";

function MessageBubble({
    role,
    content,
    intent,
    sources = [],
    handoff = false,
}) {
    const isUser = role === "user";
    const isError = role === "error";
    const isAssistant = !isUser && !isError;

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
                        <>
                            <ReactMarkdown>
                                {content}
                            </ReactMarkdown>

                            {/* ------------------------------------------------
                                CUSTOMER-FACING SOURCES
                            ------------------------------------------------- */}

                            {sources.length > 0 && (
                                <div className="message-sources">
                                    <div className="sources-title">
                                        Sources
                                    </div>

                                    <div className="sources-list">
                                        {sources.map(
                                            (source, index) => (
                                                <div
                                                    className="source-card"
                                                    key={`${source.filename}-${source.heading}-${index}`}
                                                >
                                                    <div className="source-icon">
                                                        DOC
                                                    </div>

                                                    <div className="source-details">
                                                        <div className="source-filename">
                                                            {
                                                                source.filename
                                                            }
                                                        </div>

                                                        {source.heading && (
                                                            <div className="source-heading">
                                                                {
                                                                    source.heading
                                                                }
                                                            </div>
                                                        )}
                                                    </div>
                                                </div>
                                            )
                                        )}
                                    </div>
                                </div>
                            )}

                            {/* ------------------------------------------------
                                HANDOFF NOTICE
                            ------------------------------------------------- */}

                            {handoff && (
                                <div className="handoff-notice">
                                    <strong>
                                        Human assistance recommended
                                    </strong>

                                    <span>
                                        Please contact customer support
                                        for further clarification.
                                    </span>
                                </div>
                            )}
                        </>
                    )}
                </div>
            </div>
        </div>
    );
}

export default MessageBubble;