const API_BASE_URL =
    "http://127.0.0.1:8000";


export async function sendMessage(
    sessionId,
    message
) {
    const response = await fetch(
        `${API_BASE_URL}/api/chat`,
        {
            method: "POST",

            headers: {
                "Content-Type": "application/json",
            },

            body: JSON.stringify({
                session_id: sessionId,
                message,
            }),
        }
    );


    if (!response.ok) {
        let errorMessage =
            "Failed to communicate with the support agent.";

        try {
            const error =
                await response.json();

            if (error.detail) {
                errorMessage =
                    typeof error.detail === "string"
                        ? error.detail
                        : JSON.stringify(error.detail);
            }
        } catch {
            // Keep default error message.
        }

        throw new Error(
            errorMessage
        );
    }


    return response.json();
}