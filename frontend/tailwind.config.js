/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "#090D16",
        surface: "#0F1624",
        "surface-raised": "#141D2E",
        "surface-hover": "#1A253A",
        "surface-border": "#1E2B42",
        "surface-subtle": "#121A2B",
        brand: {
          50: "#EEF2FF",
          100: "#E0E7FF",
          400: "#818CF8",
          500: "#6366F1",
          600: "#4F46E5",
          700: "#4338CA",
        },
        safe: {
          DEFAULT: "#10B981",
          subtle: "rgba(16, 185, 129, 0.12)",
          border: "rgba(16, 185, 129, 0.28)",
          text: "#34D399",
        },
        warning: {
          DEFAULT: "#F59E0B",
          subtle: "rgba(245, 158, 11, 0.12)",
          border: "rgba(245, 158, 11, 0.28)",
          text: "#FBBF24",
        },
        hazard: {
          DEFAULT: "#F43F5E",
          subtle: "rgba(244, 63, 94, 0.12)",
          border: "rgba(244, 63, 94, 0.28)",
          text: "#FB7185",
        },
      },
      fontFamily: {
        sans: ["'Inter'", "-apple-system", "BlinkMacSystemFont", "'Segoe UI'", "Roboto", "sans-serif"],
        mono: ["'JetBrains Mono'", "'SF Mono'", "Menlo", "monospace"],
      },
      boxShadow: {
        subtle: "0 1px 3px rgba(0, 0, 0, 0.3), 0 1px 2px rgba(0, 0, 0, 0.2)",
        card: "0 4px 20px -2px rgba(0, 0, 0, 0.45), 0 0 0 1px rgba(255, 255, 255, 0.05)",
        "card-hover": "0 8px 30px -4px rgba(0, 0, 0, 0.6), 0 0 0 1px rgba(99, 102, 241, 0.3)",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0", transform: "translateY(4px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        fadeIn: "fadeIn 0.2s cubic-bezier(0.16, 1, 0.3, 1) forwards",
      },
    },
  },
  plugins: [],
};
