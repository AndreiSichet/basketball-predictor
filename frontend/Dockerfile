# Build with frontend/ as the context:
#
#   docker build -f frontend/Dockerfile -t basketball-frontend frontend
#
# Two stages: node builds the static bundle, nginx serves it. Node and
# node_modules do not survive into the runtime image.

FROM node:20-alpine AS build

RUN npm install -g npm@11.16.0

WORKDIR /build

# Lockfile first, so a source edit does not reinstall the dependency tree.
# npm ci (not install) installs exactly what package-lock.json pins.
COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# create-react-app inlines REACT_APP_* at BUILD time, not run time, so this
# has to be a build arg - setting it as a compose `environment:` entry on
# the running container would do nothing at all.
#
# It also has to be a URL the BROWSER can reach, not a compose service
# name: this string ends up in JavaScript executing on the user's machine,
# which cannot resolve "backend". The host port mapping is the address.
ARG REACT_APP_API_BASE=http://localhost:8080/api
ENV REACT_APP_API_BASE=$REACT_APP_API_BASE

RUN npm run build

FROM nginx:alpine

COPY --from=build /build/build /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
