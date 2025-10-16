import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './styles/final_visualization.css'
import 'jsmind/style/jsmind.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
