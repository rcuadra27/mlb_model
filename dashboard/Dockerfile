FROM nginx:alpine
COPY dist /usr/share/nginx/html
RUN printf '%s\n' \
  'server {' \
  '    listen 8080;' \
  '    location = /index.html {' \
  '        root /usr/share/nginx/html;' \
  '        add_header Cache-Control "no-cache, no-store, must-revalidate";' \
  '        add_header Pragma "no-cache";' \
  '        add_header Expires "0";' \
  '    }' \
  '    location /assets/ {' \
  '        root /usr/share/nginx/html;' \
  '        add_header Cache-Control "public, max-age=31536000, immutable";' \
  '    }' \
  '    location / {' \
  '        root /usr/share/nginx/html;' \
  '        index index.html;' \
  '        try_files $uri $uri/ /index.html;' \
  '    }' \
  '}' \
  > /etc/nginx/conf.d/default.conf
EXPOSE 8080
CMD ["nginx", "-g", "daemon off;"]
