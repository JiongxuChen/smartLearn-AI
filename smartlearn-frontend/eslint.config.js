import js from "@eslint/js";
import globals from "globals";
import react from "eslint-plugin-react";

export default [
  // 基础 JS 推荐规则
  js.configs.recommended,

  // 全局变量：浏览器 + ES2024
  {
    languageOptions: {
      globals: {
        ...globals.browser,
        ...globals.es2024,
      },
    },
  },

  // React 推荐规则
  {
    plugins: { react },
    rules: {
      ...react.configs.recommended.rules,
      "react/react-in-jsx-scope": "off",   // Vite 不需要 import React
    },
    settings: {
      react: { version: "detect" },
    },
  },

  // 忽略构建产物
  {
    ignores: ["dist/"],
  },
];
