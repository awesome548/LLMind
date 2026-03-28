import axios from 'axios';

const api = axios.create({
    baseURL: '/',
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