function OrderCard({ order }) {
    if (!order) {
        return null;
    }

    const status = order.status?.toLowerCase();

    let statusLabel = "Unknown";

    if (status === "shipped") {
        statusLabel = "Shipped";
    } else if (status === "delayed") {
        statusLabel = "Delayed";
    } else if (status === "cancelled") {
        statusLabel = "Cancelled";
    } else if (status) {
        statusLabel =
            status.charAt(0).toUpperCase() +
            status.slice(1);
    }

    return (
        <div className="order-card">

            <div className="order-card-header">

                <div className="order-icon">
                    #
                </div>

                <div>
                    <div className="order-title">
                        Order {order.order_id}
                    </div>

                    <div className="order-subtitle">
                        Order status
                    </div>
                </div>

            </div>

            <div className="order-card-body">

                <div className="order-field">

                    <span>
                        Status
                    </span>

                    <strong
                        className={`order-status ${status || ""}`}
                    >
                        {statusLabel}
                    </strong>

                </div>

                <div className="order-field">

                    <span>
                        Estimated delivery
                    </span>

                    <strong>
                        {order.delivery_estimate ||
                            "Not available"}
                    </strong>

                </div>

            </div>

        </div>
    );
}

export default OrderCard;