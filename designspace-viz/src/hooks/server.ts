// server.ts
import express from 'express';
import fetch from "node-fetch";
const app = express();

app.get("/img-proxy", async (req, res) => {
  const url = req.query.url as string;
  if (!url) return res.status(400).send("Missing url");
  const r = await fetch(url, { headers: { "User-Agent": "YourApp/1.0" } });
  if (!r.ok) return res.sendStatus(r.status);
  res.set("Content-Type", r.headers.get("content-type") || "image/jpeg");
  r.body.pipe(res);
});

app.listen(3001);