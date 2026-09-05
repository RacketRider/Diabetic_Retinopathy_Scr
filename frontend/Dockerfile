# Multi-stage build for production frontend
FROM node:22-slim AS build

WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci --ignore-scripts 2>/dev/null || npm install --ignore-scripts
COPY . .
RUN npm run build

# Serve with nginx
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html

# nginx config that proxies /api to backend
RUN printf 'server {\n\
    listen 80;\n\
    root /usr/share/nginx/html;\n\
    index index.html;\n\
\n\
    # SPA fallback\n\
    location / {\n\
        try_files $uri $uri/ /index.html;\n\
    }\n\
\n\
    # Proxy API calls to backend\n\
    location /api/ {\n\
        proxy_pass http://backend:8000;\n\
        proxy_set_header Host $host;\n\
        proxy_set_header X-Real-IP $remote_addr;\n\
        client_max_body_size 15M;\n\
    }\n\
\n\
    # Security headers\n\
    add_header X-Content-Type-Options nosniff always;\n\
    add_header X-Frame-Options DENY always;\n\
    add_header Referrer-Policy strict-origin-when-cross-origin always;\n\
}\n' > /etc/nginx/conf.d/default.conf

EXPOSE 80
