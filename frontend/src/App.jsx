import { useState } from "react";

import ChatInput from "./components/ChatInput";
import MessageBubble from "./components/MessageBubble";
import SourceCard from "./components/SourceCard";
import OrderCard from "./components/OrderCard";
import HandoffBanner from "./components/HandoffBanner";

import { sendMessage } from "./services/api";


function App() {
  const [messages, setMessages] = useState([]);

  const [input, setInput] = useState("");

  const [loading, setLoading] = useState(false);

  const [sessionId] = useState(
    () =>
      `web-${Date.now()}-${Math.random()
        .toString(36)
        .slice(2, 9)}`
  );


  async function handleSend() {
    const message = input.trim();

    if (!message || loading) {
      return;
    }

    setInput("");

    // ---------------------------------------------------------------
    // Add customer message immediately.
    // ---------------------------------------------------------------

    setMessages((previous) => [
      ...previous,
      {
        role: "user",
        content: message,
      },
    ]);

    setLoading(true);

    try {
      // -------------------------------------------------------------
      // Call backend.
      // -------------------------------------------------------------

      const result = await sendMessage(
        sessionId,
        message
      );

      // -------------------------------------------------------------
      // Store complete structured response.
      // -------------------------------------------------------------

      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",

          content: result.answer,

          sources:
            result.sources || [],

          handoff:
            result.handoff || false,

          intent:
            result.intent || "",

          order:
            result.order || null,
        },
      ]);

    } catch (error) {

      setMessages((previous) => [
        ...previous,
        {
          role: "error",

          content:
            error.message ||
            "Something went wrong. Please try again.",
        },
      ]);

    } finally {

      setLoading(false);

    }
  }


  function setSuggestion(text) {
    setInput(text);
  }


  return (
    <div className="app">

      {/* =========================================================
          HEADER
      ========================================================= */}

      <header className="header">

        <div className="brand">

          <div className="brand-mark">
            A
          </div>

          <div>
            <h1>
              Aster & Row
            </h1>

            <p>
              Customer Support
            </p>
          </div>

        </div>


        <div className="status">

          <span className="status-dot" />

          Online

        </div>

      </header>


      {/* =========================================================
          CHAT AREA
      ========================================================= */}

      <main className="chat-container">

        {/* =======================================================
            WELCOME SCREEN
        ======================================================= */}

        {messages.length === 0 && (

          <section className="welcome">

            <div className="welcome-icon">
              A
            </div>

            <h2>
              How can we help?
            </h2>

            <p>
              Ask about returns, shipping,
              warranty, orders, or other
              Aster & Row policies.
            </p>


            <div className="suggestions">

              <button
                onClick={() =>
                  setSuggestion(
                    "How long does a regular customer have to return an unused backpack?"
                  )
                }
              >
                Return policy
              </button>


              <button
                onClick={() =>
                  setSuggestion(
                    "What is the status of ORD-1003?"
                  )
                }
              >
                Check an order
              </button>


              <button
                onClick={() =>
                  setSuggestion(
                    "What is the warranty period?"
                  )
                }
              >
                Warranty
              </button>

            </div>

          </section>

        )}


        {/* =======================================================
            MESSAGES
        ======================================================= */}

        <section className="messages">

          {messages.map(
            (message, index) => {

              const isAssistant =
                message.role === "assistant";

              const isOrder =
                isAssistant &&
                message.intent === "order";

              return (

                <div
                  key={index}
                  className="message-container"
                >

                  {/* ------------------------------------------------
                      Main message
                  ------------------------------------------------ */}

                  <MessageBubble
                    role={message.role}
                    content={message.content}
                    intent={message.intent}
                  />


                  {/* ------------------------------------------------
                      Structured order card
                  ------------------------------------------------ */}

                  {isOrder &&
                    message.order && (

                      <div className="assistant-extra">

                        <OrderCard
                          order={message.order}
                        />

                      </div>

                    )}


                  {/* ------------------------------------------------
                      Knowledge-base sources
                  ------------------------------------------------ */}

                  {isAssistant &&
                    !isOrder &&
                    message.sources?.length > 0 && (

                      <div className="assistant-extra">

                        <div className="sources-title">
                          Sources
                        </div>

                        <div className="sources">

                          {message.sources.map(
                            (
                              source,
                              sourceIndex
                            ) => (

                              <SourceCard
                                key={
                                  sourceIndex
                                }
                                source={source}
                              />

                            )
                          )}

                        </div>

                      </div>

                    )}


                  {/* ------------------------------------------------
                      Human handoff
                  ------------------------------------------------ */}

                  {isAssistant &&
                    message.handoff && (

                      <div className="assistant-extra">

                        <HandoffBanner />

                      </div>

                    )}

                </div>

              );
            }
          )}


          {/* =======================================================
              LOADING INDICATOR
          ======================================================= */}

          {loading && (

            <div className="message-row assistant">

              <div className="message-avatar">
                A&R
              </div>

              <div className="message-content">

                <div className="message-label">
                  Aster & Row Support
                </div>

                <div className="message-bubble typing">

                  <span />
                  <span />
                  <span />

                </div>

              </div>

            </div>

          )}

        </section>

      </main>


      {/* =========================================================
          INPUT
      ========================================================= */}

      <ChatInput
        value={input}
        onChange={setInput}
        onSend={handleSend}
        loading={loading}
      />

    </div>
  );
}

export default App;