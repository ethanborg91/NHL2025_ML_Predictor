import "./globals.css";

export const metadata = {
  title: "NHL 25-26 Predictor",
  description: "Predicts the standings for the upcoming NHL season",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className="bg-white text-gray-50">{children}</body>
    </html>
  );
}
