"use client";

import { useEffect, useState } from "react";
import { Card, CardBody } from "@heroui/card";
import { Button } from "@heroui/button";
import { Image } from "@heroui/image";
import Link from "next/link";
import { Chip } from "@heroui/chip";
import { useRouter } from "next/navigation";

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
  const [inputImage, setInputImage] = useState<string | null>(null);
  const [results, setResults] = useState<SearchResultItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const storedData = sessionStorage.getItem("searchResultData");

    if (storedData) {
        try {
            const parsedData = JSON.parse(storedData);
            setInputImage(parsedData.inputImage);
            setResults(parsedData.results);
        } catch (e) {
            console.error("Failed to parse search results", e);
        }
    } else {
        // perlu buat re-direct kalo lgsg akses jalurnya
        router.push("/book-collection"); 
    }
    setLoading(false);
  }, [router]);

  if (loading) return <div className="min-h-screen flex items-center justify-center">Loading results...</div>;

  return (
    <section className="w-full py-8 min-h-screen">
      <div className="w-full px-6 md:px-12 max-w-[1600px] mx-auto">
        
        <div className="mb-8">
            <h1 className="text-3xl font-bold">Hasil Pencarian Visual</h1>
            <p className="text-default-500 mt-2">Menampilkan buku yang memiliki kemiripan visual dengan gambar yang Anda unggah.</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Gambar Input*/}
          <div className="lg:col-span-3">
             <div className="sticky top-24">
                <div className="mb-4 flex items-center gap-2">
                    <Chip color="primary" variant="dot" className="font-bold">Target Pencarian</Chip>
                </div>
                
                <Card className="border-none bg-background/60 dark:bg-default-100/50 w-full shadow-md">
                    <CardBody className="p-4">
                        <div className="bg-black/5 dark:bg-white/5 rounded-xl p-4 flex items-center justify-center aspect-[3/4] mb-4">
                            {inputImage ? (
                                <Image
                                    alt="Uploaded Image"
                                    className="w-full h-full object-contain"
                                    src={inputImage}
                                    width="100%"
                                    height="100%"
                                    removeWrapper
                                />
                            ) : (
                                <div className="text-small text-default-400">No Image</div>
                            )}
                        </div>
                        <h3 className="text-lg font-bold text-center">Gambar Input</h3>
                    </CardBody>
                </Card>

             </div>
          </div>

          {/* Hasil Search */}
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
                                                <span className="text-[10px] uppercase font-bold tracking-widest text-default-500 mb-1">
                                                    Similarity Score: {item.score}
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