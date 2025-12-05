"use client";

import { useEffect, useState } from "react";
import { Card, CardBody } from "@heroui/card";
import { Image } from "@heroui/image";
import { Button } from "@heroui/button";
import Link from "next/link";

interface RecommendationViewProps {
  currentBookId: string;
}

const API_BASE_URL = "http://localhost:8000";

export const BookRecommendationView = ({ currentBookId }: RecommendationViewProps) => {
  const [recommendations, setRecommendations] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchRecommendations = async () => {
      if (!currentBookId) return;

      setLoading(true);
      try {
        const response = await fetch(`${API_BASE_URL}/api/books/${currentBookId}/recommendation`);
        
        if (!response.ok) {
          throw new Error(`Error: ${response.status}`);
        }

        const data = await response.json();
        setRecommendations(data.recommendations || []);

      } catch (error) {
        console.error("Failed to fetch recommendations", error);
      } finally {
        setLoading(false);
      }
    };

    fetchRecommendations();
  }, [currentBookId]);

  return (
    <div className="animate-in fade-in slide-in-from-right-4 duration-500">
      <div className="text-center mb-10">
        <h2 className="text-2xl font-bold">Rekomendasi Buku</h2>
        <p className="text-default-500 text-sm">Berdasarkan tingkat kemiripan konten</p>
      </div>

      {loading ? (
        <div className="flex justify-center p-10">
          <p className="text-default-400">Loading recommendations...</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 2xl:grid-cols-3 gap-6 w-full">
          {recommendations.length > 0 ? (
            recommendations.map((item) => {
              let cleanCover = item.cover ? item.cover.replace(/\\/g, '/') : 'placeholder.jpg';
              const imageUrl = cleanCover.startsWith('http') || cleanCover.startsWith('/') 
                ? cleanCover 
                : `/${cleanCover}`;

              return (
                <Card
                  key={item.id}
                  isBlurred
                  className="border-none bg-background/60 dark:bg-default-100/50 w-full hover:bg-content2/50 transition-all duration-300"
                  shadow="sm"
                >
                  <CardBody className="p-0 overflow-hidden">
                    <div className="flex flex-row h-full">
                      
                      <div className="w-32 sm:w-40 h-full relative flex-shrink-0">
                        <Image
                          alt={item.title}
                          className="object-cover w-full h-full rounded-none"
                          src={imageUrl}
                          radius="none"
                          width="100%"
                          height="100%"
                          removeWrapper
                          fallbackSrc="https://via.placeholder.com/160x200?text=No+Cover"
                        />
                      </div>

                      <div className="flex flex-col justify-between p-4 flex-grow">
                        <div>
                          <div className="flex justify-between items-center w-full mb-1">
                            <span className="text-[10px] uppercase font-bold tracking-widest text-default-500">
                              #{item.id}
                            </span>
                            
                            <div className="flex items-center gap-1 bg-primary/10 px-2 py-1 rounded-full">
                              <span className="text-[10px] font-bold text-primary">
                                Similarity Score: {item.similarity.toFixed(4)}
                              </span>
                            </div>
                          </div>

                          <Link href={`/book-collection/${item.id}`} className="group">
                            <h3 className="text-lg sm:text-xl font-bold text-foreground leading-tight line-clamp-2 group-hover:text-primary transition-colors" title={item.title}>
                              {item.title}
                            </h3>
                          </Link>
                          
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
                            Baca Detail
                          </Button>
                        </div>
                      </div>
                    </div>
                  </CardBody>
                </Card>
              );
            })
          ) : (
            <div className="col-span-full text-center p-10">
              <p className="text-default-400">Belum ada rekomendasi yang mirip.</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};