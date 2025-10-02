/** @type {import('next').NextConfig} */
const path = require('path')
const nextConfig = {
  reactStrictMode: true,
  output: 'standalone',
  // Silence workspace root detection warning by pointing tracing to project root (tgTrax)
  outputFileTracingRoot: path.resolve(__dirname, '..')
}

module.exports = nextConfig
