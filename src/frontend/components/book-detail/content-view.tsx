"use client";

import { Image } from "@heroui/image";

export const BookContentView = ({ book }: { book: any }) => {
  let cleanCover = book.cover ? book.cover.replace(/\\/g, '/') : 'placeholder.jpg';
  const imageUrl = cleanCover.startsWith('http') || cleanCover.startsWith('/') 
    ? cleanCover 
    : `/${cleanCover}`;

  return (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 w-full">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold mb-2">Book Content's</h1>
      </div>

      <div className="flex justify-center mb-10 w-full">
        <div className="relative max-w-lg w-full flex justify-center">
            <Image
            isBlurred
            alt={book.title}
            src={imageUrl}
            className="w-auto h-auto max-w-full max-h-[500px] object-contain rounded-2xl shadow-xl z-10"
            />
        </div>
      </div>

      <div className="prose dark:prose-invert max-w-none text-default-600 text-lg leading-relaxed text-justify px-2 md:px-0">
        {book.txt ? (
            book.txt.split('\n').map((paragraph: string, idx: number) => (
                <p key={idx} className="mb-4">{paragraph}</p>
            ))
        ) : (
            <p className="text-center italic text-default-400">Konten tidak tersedia.</p>
        )}
      </div>
    </div>
  );
};