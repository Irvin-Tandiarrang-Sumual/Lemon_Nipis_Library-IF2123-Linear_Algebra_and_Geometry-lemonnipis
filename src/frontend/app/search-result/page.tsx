"use client";

import { useEffect, useState } from "react";
import { Card, CardBody } from "@heroui/card";
import { Button } from "@heroui/button";
import { Image } from "@heroui/image";
import Link from "next/link";
import { Chip } from "@heroui/chip";
import { useRouter, useSearchParams } from "next/navigation";

interface SearchResultItem {
    id: string;
    title: string;
    cover: string;
    score?: number;
    file_name?: string;
}

const getImageUrl = (coverPath: string) => {
    if (!coverPath) return 'placeholder.jpg';
    let cleanPath = coverPath.replace(/\\/g, '/');
    if (cleanPath.startsWith('http')) return cleanPath;
    return cleanPath.startsWith('/') ? cleanPath : `/${cleanPath}`;
};

export default function SearchResultPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  
  const [inputType, setInputType] = useState<"image" | "text">("image");
  const [inputContent, setInputContent] = useState<string | null>(null);
  const [inputName, setInputName] = useState<string>("");
  
  const [results, setResults] = useState<SearchResultItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const storedData = sessionStorage.getItem("searchResultData");

    if (storedData) {
        try {
            const parsedData = JSON.parse(storedData);
            
            setInputType(parsedData.inputType || "image"); 
            setInputContent(parsedData.inputContent || parsedData.inputImage);
            setInputName(parsedData.inputName || "");
            
            setResults(parsedData.results);
        } catch (e) {
            console.error("Failed to parse search results", e);
        }
    } else {

    }
    setLoading(false);
  }, [router, searchParams]);

  if (loading) return <div className="min-h-screen flex items-center justify-center">Loading results...</div>;

  return (
    <section className="w-full py-8 min-h-screen">
      <div className="w-full px-6 md:px-12 max-w-[1600px] mx-auto">
        
        <div className="mb-8">
            <h1 className="text-3xl font-bold">Hasil Pencarian {inputType === "text" ? "Konten (Dokumen)" : "Visual (Gambar)"}</h1>
            <p className="text-default-500 mt-2">
                {inputType === "text" 
                 ? "Menampilkan buku yang memiliki kemiripan isi konten dengan dokumen teks yang Anda unggah."
                 : "Menampilkan buku yang memiliki kemiripan visual dengan gambar yang Anda unggah."}
            </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          <div className="lg:col-span-3">
             <div className="sticky top-24">
                <div className="mb-4 flex items-center gap-2">
                    <Chip color="primary" variant="dot" className="font-bold">Target Pencarian</Chip>
                </div>
                
                <Card className="border-none bg-background/60 dark:bg-default-100/50 w-full shadow-md">
                    <CardBody className="p-4">
                        
                        <div className="bg-black/5 dark:bg-white/5 rounded-xl p-4 mb-4 overflow-hidden">
                            {inputType === "image" ? (
                                <div className="flex items-center justify-center aspect-[3/4]">
                                    {inputContent ? (
                                        <Image
                                            alt="Uploaded Image"
                                            className="w-full h-full object-contain"
                                            src={inputContent}
                                            width="100%"
                                            height="100%"
                                            removeWrapper
                                        />
                                    ) : <div className="text-small text-default-400">No Image</div>}
                                </div>
                            ) : (
                                <div className="flex flex-col h-[300px]">
                                    <div className="flex items-center gap-2 mb-3 text-primary border-b border-default-200 pb-2">
                                        <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                        </svg>
                                        <span className="font-semibold text-sm truncate" title={inputName}>{inputName}</span>
                                    </div>
                                    <div className="flex-grow overflow-y-auto text-xs font-mono text-default-600 bg-white/50 dark:bg-black/10 p-2 rounded whitespace-pre-wrap">
                                        {inputContent && inputContent.length > 500 
                                            ? inputContent.substring(0, 500) + "..." 
                                            : inputContent || "Tidak ada konten text."}
                                    </div>
                                </div>
                            )}
                        </div>

                        <h3 className="text-lg font-bold text-center">
                            {inputType === "image" ? "Gambar Input" : "Isi Dokumen"}
                        </h3>
                    </CardBody>
                </Card>

             </div>
          </div>

          <div className="lg:col-span-9">
             <div className="mb-4 flex items-center justify-between">
                <h2 className="text-xl font-bold">
                    {results.length > 0 ? `Ditemukan ${results.length} Buku Serupa` : "Tidak ada hasil ditemukan"}
                </h2>
             </div>

             <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                {results.map((item) => {
                    const coverUrl = getImageUrl(item.cover);
                    
                    return (
                    <Card
                        key={item.id}
                        isBlurred
                        className="border-none bg-background/60 dark:bg-default-100/50 w-full hover:bg-content2/50 transition-all duration-300"
                        shadow="sm"
                    >
                        <CardBody className="p-0 overflow-hidden">
                            <div className="flex flex-row h-full min-h-[200px]">
                                
                                <div className="w-32 sm:w-40 relative flex-shrink-0 bg-black/5 dark:bg-white/5 p-2 flex items-center justify-center">
                                    <Image
                                        alt={item.title}
                                        className="object-contain w-full h-full rounded-md"
                                        src={coverUrl}
                                        radius="none"
                                        width="100%"
                                        height="100%"
                                        removeWrapper
                                    />
                                </div>

                                <div className="flex flex-col justify-between p-4 flex-grow">
                                    <div>
                                        <div className="flex justify-between items-start w-full">
                                            <div className="flex flex-col pr-2 w-full">
                                                <span className={`text-[10px] uppercase font-bold tracking-widest mb-1 ${inputType === 'text' ? 'text-success' : 'text-default-500'}`}>
                                                    {inputType === 'text' 
                                                        ? `Similarity: ${item.score!.toFixed(4)}` 
                                                        : `Score: ${item.score?.toFixed(4) || '-'}`}
                                                </span>
                                                <Link href={`/book-collection/${item.id}`} className="group block">
                                                    <h3 className="text-lg sm:text-xl font-bold text-foreground leading-tight line-clamp-2 group-hover:text-primary transition-colors" title={item.title}>
                                                        {item.title}
                                                    </h3>
                                                </Link>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="mt-4 w-full flex justify-end">
                                        <Button
                                            as={Link}
                                            href={`/book-collection/${item.id}`}
                                            className="font-medium w-full sm:w-auto px-6"
                                            color="primary"
                                            radius="full"
                                            size="sm"
                                            variant="flat"
                                        >
                                            Lihat Buku
                                        </Button>
                                    </div>
                                </div>
                            </div>
                        </CardBody>
                    </Card>
                )})}
             </div>
          </div>

        </div>
      </div>
    </section>
  );
}