FROM node:18-alpine AS base

# Install dependencies only when needed
FROM base AS deps
RUN apk add --no-cache libc6-compat
WORKDIR /app

COPY package.json pnpm-lock.yaml* ./

# Installer pnpm via npm
RUN npm install -g pnpm

# Installer les dépendances avec pnpm
RUN pnpm install --frozen-lockfile

# Rebuild the source code only when needed
FROM base AS builder
WORKDIR /app

# Re-Installer pnpm au cas où il serait nécessaire dans cette étape
RUN npm install -g pnpm

COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Vérifier si pnpm est installé correctement
RUN pnpm --version  # Vérifier si pnpm est installé correctement

# Exécuter les commandes avec pnpm
RUN pnpm run prisma:generate
RUN pnpm run build

# Production image, copy all the files and run next
FROM base AS runner
WORKDIR /app

ENV NODE_ENV production

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public

# Automatically leverage output traces to reduce image size
# https://nextjs.org/docs/advanced-features/output-file-tracing
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

ENV PORT 3000

CMD ["node", "server.js"]
