import { useState } from "react";
import { sendMessage } from "../services/api";

export default function Chat() {
  const [msg, setMsg] = useState("");
  const [response, setResponse] = useState("");

  const handleSend = async () => {
    const res = await sendMessage(msg);
    setResponse(res.response);
  };

  return (
    <div>
      <h1>Chat UI</h1>
      <input value={msg} onChange={(e) => setMsg(e.target.value)} />
      <button onClick={handleSend}>Send</button>
      <p>{response}</p>
    </div>
  );
}
