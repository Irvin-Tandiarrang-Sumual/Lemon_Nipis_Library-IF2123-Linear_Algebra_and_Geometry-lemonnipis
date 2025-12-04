import { notFound } from "next/navigation";
import { BookDetailWrapper } from "@/components/book-detail/detail-wrapper";

// const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const API_BASE_URL = "http://localhost:8000";

async function getBookById(id: string) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/books/${id}/content`, {
      cache: "no-store",
    });

    if (!res.ok) {
      if (res.status === 404) return null;
      throw new Error("Failed to fetch book");
    }
    return res.json();
  } catch (error) {
    console.error("Error fetching book:", error);
    return null;
  }
}

// PERUBAHAN 1: Ubah tipe params menjadi Promise<{ id: string }>
export default async function BookDetailPage({ params }: { params: Promise<{ id: string }> }) {
  
  // PERUBAHAN 2: Await params sebelum destructuring id
  const { id } = await params; 

  // Gunakan id yang sudah di-resolve
  const book = await getBookById(id);

  if (!book) {
    notFound();
  }

  return (
    <BookDetailWrapper book={book} />
  );
}