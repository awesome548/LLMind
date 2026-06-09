import axios from 'axios';

// Call the backend directly rather than through Next.js rewrites. The dev proxy
// fails to deliver responses for long-running LLM requests (generation can take
// 50s+ with a local model), leaving the UI stuck. A direct connection (with CORS
// enabled on the backend) handles long requests reliably.
// Override with NEXT_PUBLIC_API_BASE_URL; empty string falls back to the proxy.
const baseURL =
    process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:8000';

const api = axios.create({
    baseURL,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Add interceptors for Auth tokens if needed
// api.interceptors.request.use((config) => {
//     const token = localStorage.getItem('auth-token');
//     if (token) {
//         config.headers.Authorization = `Bearer ${token}`;
//     }
//     return config;
// });

export default api;