# KneeXpert frontend — Vite + React dev server
FROM node:20-slim

WORKDIR /app

# Install deps first (better layer caching)
COPY package.json package-lock.json ./
RUN npm install

# App source
COPY . .

EXPOSE 8080

# Dev server; --host exposes it outside the container
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "8080"]
