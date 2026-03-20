import axios from "axios";

const API = "http://localhost:8000";

export const sendMessage = async (message) => {
  const res = await axios.post(`${API}/chat`, { message });
  return res.data;
};
