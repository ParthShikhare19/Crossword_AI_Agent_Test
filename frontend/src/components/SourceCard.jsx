function SourceCard({ source }) {
    return (
        <div className="source-card">
            <div className="source-icon">
                DOC
            </div>

            <div className="source-details">
                <strong>{source.filename}</strong>

                <span>{source.heading}</span>
            </div>
        </div>
    );
}

export default SourceCard;