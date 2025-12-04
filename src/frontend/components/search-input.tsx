"use client";

import { useRef, useState, useEffect } from "react";
import { Input } from "@heroui/input";
import { SearchIcon, CameraIcon, CloseIcon } from "@/components/icons"; 
import { useRouter, useSearchParams } from "next/navigation";

// Spinner kecil untuk indikator loading
const LoadingSpinner = () => (
  <svg className="animate-spin h-4 w-4 text-primary" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
  </svg>
);

const fileToBase64 = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = (error) => reject(error);
  });
};

export const CustomSearchInput = ({ className }: { className?: string }) => {
  const router = useRouter();
  const searchParams = useSearchParams();
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const currentQuery = searchParams.get("q") || "";
  const API_BASE_URL = "http://localhost:8000";

  useEffect(() => {
    router.prefetch("/search-result");
  }, [router]);

  const handleAutoSearchImage = async (file: File) => {
    setIsLoading(true); 
    const formData = new FormData();
    formData.append("file", file);

    try {
      const base64Promise = fileToBase64(file);
      const apiPromise = fetch(`${API_BASE_URL}/api/books/search-by-image`, {
        method: "POST",
        body: formData,
      });

      const [base64Input, res] = await Promise.all([base64Promise, apiPromise]);

      if (!res.ok) throw new Error("Gagal mencari gambar");
      const data = await res.json();

      const searchData = {
        inputImage: base64Input, 
        results: data.query_results,
        timestamp: new Date().getTime()
      };

      sessionStorage.setItem("searchResultData", JSON.stringify(searchData));
      
      router.push("/search-result");

    } catch (error) {
      console.error(error);
      alert("Gagal memproses gambar atau koneksi ke server bermasalah.");
      setSelectedImage(null);
      setIsLoading(false);
    } 
  };

  const handleTextSearch = (query: string) => {
    if (!query.trim()) {
        router.push("/book-collection");
        return;
    }
    router.push(`/book-collection?q=${encodeURIComponent(query)}&page=1`);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      handleTextSearch(e.currentTarget.value);
    }
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const imageUrl = URL.createObjectURL(file);
      setSelectedImage(imageUrl);
      
      setTimeout(() => {
          handleAutoSearchImage(file);
      }, 100);

      event.target.value = "";
    }
  };

  const handleCameraClick = () => {
    fileInputRef.current?.click();
  };

  const handleClearImage = () => {
    setSelectedImage(null);
    setIsLoading(false);
  };

  return (
    <div className={className}>
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        className="hidden"
        accept="image/*"
      />

      <Input
        aria-label="Search"
        isDisabled={isLoading} 
        classNames={{
          inputWrapper: "bg-default-100",
          input: "text-sm",
          base: "w-full h-10",
        }}
        defaultValue={currentQuery}
        onKeyDown={handleKeyDown}
        startContent={
          selectedImage ? (
            <div className="flex items-center gap-2 mr-1">
              <div className="relative w-8 h-8 rounded overflow-hidden border border-default-300">
                <img 
                    src={selectedImage} 
                    alt="Preview" 
                    className={`w-full h-full object-cover transition-opacity ${isLoading ? 'opacity-50' : 'opacity-100'}`} 
                />
                {isLoading && (
                    <div className="absolute inset-0 flex items-center justify-center bg-black/10">
                        <LoadingSpinner />
                    </div>
                )}
              </div>
              
              {!isLoading && (
                  <button
                    onClick={handleClearImage}
                    className="text-default-400 hover:text-danger p-0.5 rounded-full hover:bg-default-200 transition-colors"
                    type="button"
                  >
                    <CloseIcon className="w-4 h-4" />
                  </button>
              )}
            </div>
          ) : (
            <SearchIcon className="text-base text-default-400 pointer-events-none flex-shrink-0" />
          )
        }
        endContent={
          <button
            className="focus:outline-none hover:opacity-70 text-default-400 hover:text-primary transition-colors"
            type="button"
            onClick={handleCameraClick}
            disabled={isLoading}
            title="Upload Image"
          >
            <CameraIcon className="flex-shrink-0" />
          </button>
        }
        labelPlacement="outside"
        placeholder={isLoading ? "Sedang menganalisis gambar..." : selectedImage ? "Mencari..." : "Search title or import image..."}
        type="search"
      />
    </div>
  );
};