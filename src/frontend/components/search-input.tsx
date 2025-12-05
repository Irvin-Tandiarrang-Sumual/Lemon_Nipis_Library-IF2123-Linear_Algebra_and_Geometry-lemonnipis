"use client";

import { useRef, useState, useEffect } from "react";
import { Input } from "@heroui/input";
import { SearchIcon, CameraIcon, CloseIcon } from "@/components/icons"; 
import { useRouter, useSearchParams } from "next/navigation";

const DocumentIcon = ({ className }: { className?: string }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
  </svg>
);

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

const readTextFile = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsText(file);
    });
};

export const CustomSearchInput = ({ className }: { className?: string }) => {
  const router = useRouter();
  const searchParams = useSearchParams();
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const currentQuery = searchParams.get("q") || "";
  
  const [inputValue, setInputValue] = useState(currentQuery);

  const [previewSource, setPreviewSource] = useState<string | null>(null);
  const [fileType, setFileType] = useState<"image" | "text" | null>(null);
  const [fileName, setFileName] = useState<string>("");
  const [isLoading, setIsLoading] = useState(false);

  const API_BASE_URL = "http://localhost:8000";

  useEffect(() => {
    setInputValue(currentQuery);
  }, [currentQuery]);

  useEffect(() => {
    router.prefetch("/search-result");
  }, [router]);

  const handleAutoSearchFile = async (file: File, type: "image" | "text") => {
    setIsLoading(true); 
    const formData = new FormData();
    formData.append("file", file);

    try {
      let apiPromise;
      let inputPreview;

      if (type === "image") {
        inputPreview = await fileToBase64(file);
        apiPromise = fetch(`${API_BASE_URL}/api/books/search-by-image`, {
            method: "POST",
            body: formData,
        });
      } else {
        inputPreview = await readTextFile(file); 
        apiPromise = fetch(`${API_BASE_URL}/api/books/search-by-document`, {
            method: "POST",
            body: formData,
        });
      }

      const res = await apiPromise;

      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(`Server Error: ${res.status} - ${errorText}`);
      }
      
      const data = await res.json();

      const normalizedResults = data.query_results.map((item: any) => ({
         id: item.id,
         title: item.title,
         cover: item.cover,
         score: item.score !== undefined ? item.score : item.similarity, 
         file_name: item.file_name
      }));

      const searchData = {
        inputType: type, 
        inputContent: inputPreview, 
        inputName: file.name,
        results: normalizedResults,
        timestamp: new Date().getTime()
      };

      sessionStorage.setItem("searchResultData", JSON.stringify(searchData));
      router.push(`/search-result?t=${Date.now()}`);

    } catch (error) {
      console.error(error);
      alert(`Gagal memproses ${type === 'image' ? 'gambar' : 'dokumen'}. Pastikan backend menyala.`);
      handleClearFile();
    } finally {
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
        handleTextSearch(inputValue);
    }
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    const inputElement = event.target;

    if (file) {
      if (file.type.startsWith("image/")) {
          const imageUrl = URL.createObjectURL(file);
          setPreviewSource(imageUrl);
          setFileType("image");
          setFileName(file.name);
          setTimeout(() => handleAutoSearchFile(file, "image"), 100);
      } else if (file.type === "text/plain" || file.name.endsWith(".txt")) {
          setPreviewSource(null);
          setFileType("text");
          setFileName(file.name);
          setTimeout(() => handleAutoSearchFile(file, "text"), 100);
      } else {
          alert("Format file tidak didukung. Harap gunakan Gambar atau .txt");
          return;
      }
      
      inputElement.value = "";
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleClearFile = () => {
    setPreviewSource(null);
    setFileType(null);
    setFileName("");
    setIsLoading(false);
  };

  const renderStartContent = () => {
    if (!fileType) return <SearchIcon className="text-base text-default-400 pointer-events-none flex-shrink-0" />;

    return (
        <div className="flex items-center gap-2 mr-1 max-w-[120px]">
            <div className={`relative flex items-center justify-center rounded overflow-hidden border border-default-300 ${fileType === 'image' ? 'w-8 h-8' : 'w-auto h-8 px-2 bg-default-200'}`}>
                
                {fileType === "image" && previewSource && (
                    <img 
                        src={previewSource} 
                        alt="Preview" 
                        className={`w-full h-full object-cover ${isLoading ? 'opacity-50' : 'opacity-100'}`} 
                    />
                )}

                {fileType === "text" && (
                    <div className="flex items-center gap-1 text-xs text-default-600 font-medium">
                        <DocumentIcon className="w-4 h-4" />
                        <span className="truncate max-w-[60px]">{fileName}</span>
                    </div>
                )}

                {isLoading && (
                    <div className="absolute inset-0 flex items-center justify-center bg-black/10">
                        <LoadingSpinner />
                    </div>
                )}
            </div>
            
            {!isLoading && (
                <button
                    onClick={handleClearFile}
                    className="text-default-400 hover:text-danger p-0.5 rounded-full hover:bg-default-200 transition-colors"
                    type="button"
                >
                    <CloseIcon className="w-4 h-4" />
                </button>
            )}
        </div>
    );
  };

  return (
    <div className={className}>
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        className="hidden"
        accept="image/*, .txt, text/plain"
      />

      <Input
        aria-label="Search"
        isDisabled={isLoading} 
        classNames={{
          inputWrapper: "bg-default-100",
          input: "text-sm",
          base: "w-full h-10",
        }}
        value={inputValue}
        onValueChange={setInputValue} 
        onKeyDown={handleKeyDown}
        startContent={renderStartContent()}
        endContent={
          <button
            className="focus:outline-none hover:opacity-70 text-default-400 hover:text-primary transition-colors"
            type="button"
            onClick={handleUploadClick}
            disabled={isLoading}
            title="Upload Image or Document"
          >
            <CameraIcon className="flex-shrink-0" />
          </button>
        }
        labelPlacement="outside"
        placeholder={isLoading ? "Sedang memproses..." : fileType ? "Mencari..." : "Search title, image or .txt..."}
        type="search"
      />
    </div>
  );
};